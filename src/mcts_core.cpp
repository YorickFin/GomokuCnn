// 五子棋 MCTS C++ 核心（pybind11 绑定）。
//
// 语义与 src/mcts.py 的 Python 数组化树实现逐操作对齐：
// - 扁平 SoA 树（children/priors/clean_priors/visits/virt/w_sum/...）；
// - PUCT 选择（非法着法 -1e9 掩码、argmax 取首个最大值）；
// - 批量推理：虚拟访问去相关 → 攒批去重 → 回调 Python 推理 →
//   展开 + 撤销虚拟访问 + 回传；
// - 终局节点即时回传（不走网络），满盘平局优先于五连判定；
// - 根节点干净先验 + Dirichlet 重加噪（噪声由 Python 回调提供，
//   保证与 Python 后端 RNG 消费序列一致）；
// - 跨手树复用：_reroot 子树压缩，BFS 序与统计完整保留。
//
// 边界约定：树操作与统计全部在本模块内完成；神经网络推理与
// Dirichlet 噪声采样由 Python 侧回调提供，本模块不依赖任何
// 第三方数值库。浮点上 numpy 的 SIMD 归约与标量归约可能在
// 最后一位有差异（priors 归一化、兄弟均值 ave），整数统计
// （visits/children/结构）严格一致。

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static constexpr int kBoard = 15;
static constexpr int kMoves = 225;

// 与 board.DIRECTIONS 一致：横、竖、主对角、副对角
static const int kDirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

// numpy 标量回退路径的成对求和（n<8 顺序；n<=128 八路累加；
// 否则对半递归，分割点向下取整到 8 的倍数）。
// 注：numpy 在启用 SIMD 时归约顺序可能不同，仅影响最后一位。
static double pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        size_t i = 8;
        size_t m = n - (n % 8);
        for (; i < m; i += 8) {
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) +
                     ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct MctsTree {
    double c_puct = 1.0;
    double dirichlet_alpha = 0.1;
    double noise_eps = 0.25;

    // SoA 树存储（节点 id 索引，0 为根；与 src/mcts.py 字段一一对应）
    std::vector<int32_t> children;   // cap * 225，-1 未创建
    std::vector<double> priors;      // cap * 225（加噪后）
    std::vector<double> clean_priors;  // cap * 225（未加噪，供重加噪）
    std::vector<int32_t> visits;
    std::vector<int32_t> virt;       // 虚拟访问（仅探索项）
    std::vector<double> w_sum;
    std::vector<double> val;         // 节点视角平均价值
    std::vector<double> value;       // 叶子网络/终局价值
    std::vector<uint8_t> expanded;
    std::vector<uint8_t> term_flag;
    std::vector<double> term_val;
    std::vector<int32_t> parent;
    std::vector<int32_t> move_flat;
    int64_t cap = 0;
    int64_t next_id = 1;

    MctsTree(double c_puct_, double dirichlet_alpha_, double noise_eps_)
        : c_puct(c_puct_), dirichlet_alpha(dirichlet_alpha_),
          noise_eps(noise_eps_) {}

    // ---- 树存储 ----
    void init_tree(int64_t new_cap) {
        cap = new_cap;
        children.assign((size_t)cap * kMoves, -1);
        priors.assign((size_t)cap * kMoves, 0.0);
        clean_priors.assign((size_t)cap * kMoves, 0.0);
        visits.assign((size_t)cap, 0);
        virt.assign((size_t)cap, 0);
        w_sum.assign((size_t)cap, 0.0);
        val.assign((size_t)cap, 0.0);
        value.assign((size_t)cap, 0.0);
        expanded.assign((size_t)cap, 0);
        term_flag.assign((size_t)cap, 0);
        term_val.assign((size_t)cap, 0.0);
        parent.assign((size_t)cap, -1);
        move_flat.assign((size_t)cap, -1);
        next_id = 1;
    }

    void grow(int64_t needed) {
        int64_t new_cap = std::max(cap * 2, needed);
        children.resize((size_t)new_cap * kMoves, -1);
        priors.resize((size_t)new_cap * kMoves, 0.0);
        clean_priors.resize((size_t)new_cap * kMoves, 0.0);
        visits.resize((size_t)new_cap, 0);
        virt.resize((size_t)new_cap, 0);
        w_sum.resize((size_t)new_cap, 0.0);
        val.resize((size_t)new_cap, 0.0);
        value.resize((size_t)new_cap, 0.0);
        expanded.resize((size_t)new_cap, 0);
        term_flag.resize((size_t)new_cap, 0);
        term_val.resize((size_t)new_cap, 0.0);
        parent.resize((size_t)new_cap, -1);
        move_flat.resize((size_t)new_cap, -1);
        cap = new_cap;
    }

    int64_t new_node(int64_t parent_id, int move) {
        int64_t nid = next_id;
        if (nid >= cap) grow(nid + 1);
        next_id = nid + 1;
        parent[(size_t)nid] = (int32_t)parent_id;
        move_flat[(size_t)nid] = (int32_t)move;
        return nid;
    }

    // ---- 终局判定 ----
    // 满盘返回 0.0（平局优先）；lr/lc 为刚落的点（<0 时全板扫描）。
    // 五连价值：v * (1 - stones * 3e-4)，与 board.evaluation_func 一致。
    static std::optional<double> terminal_value(const int8_t* work, int lr,
                                                int lc) {
        bool any_empty = false;
        for (int i = 0; i < kMoves; ++i) {
            if (work[i] == 0) { any_empty = true; break; }
        }
        if (!any_empty) return 0.0;

        int stones = -1;
        auto win_value = [&](int8_t v) -> double {
            if (stones < 0) {
                stones = 0;
                for (int i = 0; i < kMoves; ++i)
                    if (work[i] != 0) stones++;
            }
            return (double)v * (1.0 - (double)stones * 3e-4);
        };

        if (lr < 0) {
            // 全板扫描（与 board.winner 的扫描顺序一致，防御性路径）
            for (int i = 0; i < kBoard; ++i) {
                for (int j = 0; j < kBoard; ++j) {
                    int8_t v = work[i * kBoard + j];
                    if (v == 0) continue;
                    for (auto& d : kDirs) {
                        int cnt = 0;
                        for (int dd = 0; dd < 5; ++dd) {
                            int ni = i + d[0] * dd, nj = j + d[1] * dd;
                            if (ni >= 0 && ni < kBoard && nj >= 0 &&
                                nj < kBoard && work[ni * kBoard + nj] == v) {
                                cnt++;
                            } else {
                                break;
                            }
                        }
                        if (cnt == 5) return win_value(v);
                    }
                }
            }
            return std::nullopt;
        }

        int8_t v = work[lr * kBoard + lc];
        for (auto& d : kDirs) {
            int cnt = 1;
            int rr = lr + d[0], cc = lc + d[1];
            while (rr >= 0 && rr < kBoard && cc >= 0 && cc < kBoard &&
                   work[rr * kBoard + cc] == v) {
                cnt++;
                rr += d[0];
                cc += d[1];
            }
            rr = lr - d[0];
            cc = lc - d[1];
            while (rr >= 0 && rr < kBoard && cc >= 0 && cc < kBoard &&
                   work[rr * kBoard + cc] == v) {
                cnt++;
                rr -= d[0];
                cc -= d[1];
            }
            if (cnt >= 5) return win_value(v);
        }
        return std::nullopt;
    }

    // ---- PUCT 选择 ----
    int select(int64_t node, const int8_t* work) {
        const int32_t* ch = &children[(size_t)node * kMoves];
        const double* pr = &priors[(size_t)node * kMoves];

        // 兄弟平均价值仅用真实统计（虚拟访问不含真实价值，不得污染）
        int64_t total_real = 0;
        double wbuf[kMoves];
        int nvis = 0;
        for (int a = 0; a < kMoves; ++a) {
            int64_t c = ch[a];
            if (c >= 0) {
                int32_t v = visits[(size_t)c];
                total_real += v;
                if (v > 0) wbuf[nvis++] = w_sum[(size_t)c];
            }
        }
        double ave = pairwise_sum(wbuf, (size_t)nvis) /
                     ((double)total_real + 1e-5);
        double explore_buff = std::sqrt((double)(total_real + 1));

        double scores[kMoves];
        for (int a = 0; a < kMoves; ++a) {
            if (work[a] != 0) {
                scores[a] = -1e9;  // 非法着法（已占点）不可选
                continue;
            }
            double nv = 0.0, nvr = 0.0, w = 0.0;
            bool visited = false;
            int64_t c = ch[a];
            if (c >= 0) {
                int32_t v = visits[(size_t)c];
                nv = (double)v;
                nvr = (double)virt[(size_t)c];
                w = w_sum[(size_t)c];
                visited = v > 0;
            }
            double exploit =
                visited ? (w / std::max(nv, 1.0)) : ave;
            scores[a] = c_puct * pr[a] * explore_buff / (nv + nvr + 1.0) -
                        exploit;
        }
        int best = 0;
        double bs = scores[0];
        for (int a = 1; a < kMoves; ++a) {
            if (scores[a] > bs) {
                bs = scores[a];
                best = a;
            }
        }
        return best;
    }

    // ---- 模拟下行 ----
    void descend(const int8_t* root_board, std::vector<int32_t>& path,
                 int8_t* work, int64_t& leaf, bool& is_term) {
        std::memcpy(work, root_board, kMoves);
        int64_t node = 0;
        path.clear();
        path.push_back(0);
        while (!term_flag[(size_t)node] && expanded[(size_t)node]) {
            int a = select(node, work);
            int r = a / kBoard, c = a % kBoard;
            work[a] = 1;
            for (int i = 0; i < kMoves; ++i)
                work[i] = (int8_t)(-work[i]);  // 落子后翻转视角

            int64_t ch = children[(size_t)node * kMoves + a];
            if (ch < 0) {
                ch = new_node(node, a);
                children[(size_t)node * kMoves + a] = (int32_t)ch;
            }
            path.push_back((int32_t)ch);
            node = ch;

            if (term_flag[(size_t)node]) break;
            auto tv = terminal_value(work, r, c);
            if (tv.has_value()) {
                term_flag[(size_t)node] = 1;
                term_val[(size_t)node] = *tv;
                break;
            }
        }
        leaf = node;
        is_term = term_flag[(size_t)node] != 0;
    }

    // ---- 展开 ----
    void expand(int64_t node, const double* policy, const int8_t* board,
                bool is_root, py::function& noise_fn) {
        bool legal[kMoves];
        int n_legal = 0;
        double p[kMoves];
        for (int a = 0; a < kMoves; ++a) {
            legal[a] = board[a] == 0;
            p[a] = legal[a] ? policy[a] : 0.0;
            if (legal[a]) n_legal++;
        }
        double total = pairwise_sum(p, kMoves);
        if (total <= 0) {
            double u = 1.0 / (double)std::max(n_legal, 1);
            for (int a = 0; a < kMoves; ++a) p[a] = legal[a] ? u : 0.0;
        } else {
            for (int a = 0; a < kMoves; ++a) p[a] /= total;
        }

        // 始终保存干净先验，供树复用后根节点重新加噪使用
        double* cp = &clean_priors[(size_t)node * kMoves];
        for (int a = 0; a < kMoves; ++a) cp[a] = p[a];

        if (is_root && noise_eps > 0 && n_legal > 0) {
            F64Array eta = py::cast<F64Array>(noise_fn(py::int_(n_legal)));
            if ((int64_t)eta.size() != n_legal)
                throw std::runtime_error("noise_fn 返回长度与合法着法数不符");
            const double* ed = eta.data();
            int k = 0;
            for (int a = 0; a < kMoves; ++a) {
                if (legal[a]) {
                    p[a] = (1.0 - noise_eps) * p[a] + noise_eps * ed[k];
                    k++;
                }
            }
        }
        double* pr = &priors[(size_t)node * kMoves];
        for (int a = 0; a < kMoves; ++a) pr[a] = p[a];
        expanded[(size_t)node] = 1;
    }

    void reapply_root_noise(int64_t node, const int8_t* board,
                            py::function& noise_fn) {
        if (noise_eps <= 0) return;
        bool legal[kMoves];
        int n_legal = 0;
        for (int a = 0; a < kMoves; ++a) {
            legal[a] = board[a] == 0;
            if (legal[a]) n_legal++;
        }
        if (n_legal == 0) return;
        const double* cp = &clean_priors[(size_t)node * kMoves];
        double p[kMoves];
        for (int a = 0; a < kMoves; ++a) p[a] = cp[a];
        F64Array eta = py::cast<F64Array>(noise_fn(py::int_(n_legal)));
        if ((int64_t)eta.size() != n_legal)
            throw std::runtime_error("noise_fn 返回长度与合法着法数不符");
        const double* ed = eta.data();
        int k = 0;
        for (int a = 0; a < kMoves; ++a) {
            if (legal[a]) {
                p[a] = (1.0 - noise_eps) * p[a] + noise_eps * ed[k];
                k++;
            }
        }
        double* pr = &priors[(size_t)node * kMoves];
        for (int a = 0; a < kMoves; ++a) pr[a] = p[a];
    }

    // ---- 回传 ----
    void backprop(const std::vector<int32_t>& path, double v) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            int64_t nid = *it;
            visits[(size_t)nid] += 1;
            w_sum[(size_t)nid] += v;
            val[(size_t)nid] = w_sum[(size_t)nid] / (double)visits[(size_t)nid];
            v = -v;
        }
    }

    // ---- 批量评估 ----
    struct Pending {
        int64_t leaf;
        std::vector<int32_t> path;
        int8_t work[kMoves];
    };

    static F64Array cast_f64(const py::object& o) {
        return py::cast<F64Array>(o);
    }

    void evaluate_pending(std::vector<Pending>& pending, bool add_noise,
                          py::function& eval_fn, py::function& noise_fn) {
        // 去重：保留首次出现的叶子与棋盘（与 Python dict 语义一致）
        std::vector<int64_t> order;
        std::vector<int8_t> board_buf;
        for (auto& p : pending) {
            bool seen = false;
            for (auto lf : order) {
                if (lf == p.leaf) { seen = true; break; }
            }
            if (!seen) {
                order.push_back(p.leaf);
                board_buf.insert(board_buf.end(), p.work, p.work + kMoves);
            }
        }
        py::ssize_t nu = (py::ssize_t)order.size();

        py::array_t<int8_t> boards({nu, (py::ssize_t)kMoves});
        std::memcpy(boards.mutable_data(), board_buf.data(),
                    (size_t)nu * kMoves);

        py::tuple res = py::cast<py::tuple>(eval_fn(boards));
        F64Array values = cast_f64(res[0]);
        F64Array policies = cast_f64(res[1]);
        if (values.size() != nu || policies.size() != nu * kMoves)
            throw std::runtime_error("eval_fn 返回形状与请求批次不符");

        // 先按首现顺序逐叶展开（is_root 仅当叶子为根）
        for (py::ssize_t i = 0; i < nu; ++i) {
            int64_t leaf = order[(size_t)i];
            value[(size_t)leaf] = values.data()[i];
            expand(leaf, policies.data() + i * kMoves,
                   board_buf.data() + (size_t)i * kMoves,
                   leaf == 0 && add_noise, noise_fn);
        }
        // 再按 pending 顺序撤销虚拟访问并回传真实价值
        for (auto& p : pending) {
            double v = 0.0;
            for (size_t i = 0; i < order.size(); ++i) {
                if (order[i] == p.leaf) {
                    v = values.data()[i];
                    break;
                }
            }
            for (auto nid : p.path) virt[(size_t)nid] -= 1;
            backprop(p.path, v);
        }
    }

    // ---- 结果 ----
    py::tuple get_results(int64_t root_id) {
        const int32_t* ch = &children[(size_t)root_id * kMoves];
        double cv[kMoves];
        double total = 0.0;
        for (int a = 0; a < kMoves; ++a) {
            int64_t c = ch[a];
            double v = c >= 0 ? (double)visits[(size_t)c] : 0.0;
            cv[a] = v;
            total += v;
        }
        py::array_t<double> probs({(py::ssize_t)kBoard, (py::ssize_t)kBoard});
        double* pp = probs.mutable_data();
        if (total > 0) {
            for (int a = 0; a < kMoves; ++a) pp[a] = cv[a] / total;
        } else {
            for (int a = 0; a < kMoves; ++a) pp[a] = 0.0;
        }
        double root_value = visits[(size_t)root_id] > 0
                                ? w_sum[(size_t)root_id] /
                                      (double)visits[(size_t)root_id]
                                : 0.0;
        return py::make_tuple(root_value, probs, py::int_(0));
    }

    // ---- 跨手树复用 ----
    void reroot(int64_t old_root, int64_t num_simulations) {
        if (old_root < 0 || old_root >= next_id)
            throw std::runtime_error("cur_root 越界");
        std::vector<int64_t> order;
        std::vector<int32_t> mapping((size_t)cap, -1);
        mapping[(size_t)old_root] = 0;
        order.push_back(old_root);
        for (size_t i = 0; i < order.size(); ++i) {
            int64_t nid = order[i];
            const int32_t* ch = &children[(size_t)nid * kMoves];
            for (int a = 0; a < kMoves; ++a) {
                int64_t c = ch[a];
                if (c >= 0 && mapping[(size_t)c] < 0) {
                    mapping[(size_t)c] = (int32_t)order.size();
                    order.push_back(c);
                }
            }
        }
        int64_t n = (int64_t)order.size();

        // 迁移旧数组（move 语义避免拷贝）
        std::vector<int32_t> o_ch = std::move(children);
        std::vector<double> o_pr = std::move(priors);
        std::vector<double> o_cp = std::move(clean_priors);
        std::vector<int32_t> o_vis = std::move(visits);
        std::vector<int32_t> o_virt = std::move(virt);
        std::vector<double> o_w = std::move(w_sum);
        std::vector<double> o_val = std::move(val);
        std::vector<double> o_value = std::move(value);
        std::vector<uint8_t> o_exp = std::move(expanded);
        std::vector<uint8_t> o_tf = std::move(term_flag);
        std::vector<double> o_tv = std::move(term_val);

        init_tree(n + num_simulations + 256);

        for (int64_t new_id = 0; new_id < n; ++new_id) {
            int64_t nid = order[(size_t)new_id];
            visits[(size_t)new_id] = o_vis[(size_t)nid];
            w_sum[(size_t)new_id] = o_w[(size_t)nid];
            val[(size_t)new_id] = o_val[(size_t)nid];
            value[(size_t)new_id] = o_value[(size_t)nid];
            expanded[(size_t)new_id] = o_exp[(size_t)nid];
            term_flag[(size_t)new_id] = o_tf[(size_t)nid];
            term_val[(size_t)new_id] = o_tv[(size_t)nid];
            for (int a = 0; a < kMoves; ++a) {
                priors[(size_t)new_id * kMoves + a] =
                    o_pr[(size_t)nid * kMoves + a];
                clean_priors[(size_t)new_id * kMoves + a] =
                    o_cp[(size_t)nid * kMoves + a];
            }
            const int32_t* och = &o_ch[(size_t)nid * kMoves];
            for (int a = 0; a < kMoves; ++a) {
                int64_t c = och[a];
                if (c >= 0) {
                    int64_t nc = mapping[(size_t)c];
                    children[(size_t)new_id * kMoves + a] = (int32_t)nc;
                    parent[(size_t)nc] = (int32_t)new_id;
                    move_flat[(size_t)nc] = (int32_t)a;
                }
            }
        }
        next_id = n;
    }

    // ---- 对外主入口 ----
    py::tuple run(py::array_t<int8_t, py::array::c_style | py::array::forcecast>
                      board,
                  int64_t num_simulations, std::optional<int64_t> cur_root,
                  bool add_noise, int64_t infer_batch_size,
                  py::function eval_fn, py::function noise_fn) {
        if (board.size() != kMoves)
            throw std::runtime_error("棋盘必须为 225 个 int8 元素");
        const int8_t* root = board.data();

        if (cur_root.has_value()) {
            reroot(*cur_root, num_simulations);
        } else {
            // 每次模拟至多新建 1 个节点
            init_tree(num_simulations + 256);
        }

        // 树复用后新根已展开，需从干净先验重新注入 Dirichlet 噪声；
        // 全新树时根未展开，噪声由展开时负责，此处跳过。
        if (add_noise && expanded[0]) {
            reapply_root_noise(0, root, noise_fn);
        }

        if (infer_batch_size <= 1)
            throw std::runtime_error(
                "cpp 后端仅支持批量路径，infer_batch_size 必须 > 1");

        // 根节点必须先评估展开（等价于串行搜索的第一次模拟）
        if (!term_flag[0] && !expanded[0]) {
            auto tv = terminal_value(root, -1, -1);
            if (tv.has_value()) {
                term_flag[0] = 1;
                term_val[0] = *tv;
            } else {
                py::array_t<int8_t> b1({(py::ssize_t)1, (py::ssize_t)kMoves});
                std::memcpy(b1.mutable_data(), root, kMoves);
                py::tuple res = py::cast<py::tuple>(eval_fn(b1));
                F64Array vals = cast_f64(res[0]);
                F64Array pols = cast_f64(res[1]);
                if (vals.size() != 1 || pols.size() != kMoves)
                    throw std::runtime_error("eval_fn 返回形状与请求批次不符");
                double v = vals.data()[0];
                value[0] = v;
                expand(0, pols.data(), root, add_noise, noise_fn);
                std::vector<int32_t> rp = {0};
                backprop(rp, v);
                num_simulations -= 1;
            }
        }

        std::vector<Pending> pending;
        std::vector<int32_t> path;
        int8_t work[kMoves];
        for (int64_t i = 0; i < num_simulations; ++i) {
            int64_t leaf;
            bool is_term;
            descend(root, path, work, leaf, is_term);
            if (is_term) {
                // 终局价值立即可得，直接回传
                backprop(path, term_val[(size_t)leaf]);
                continue;
            }
            // 叶子登记并施加虚拟访问（促使同批后续模拟探索别处）
            for (auto nid : path) virt[(size_t)nid] += 1;
            Pending p;
            p.leaf = leaf;
            p.path = path;
            std::memcpy(p.work, work, kMoves);
            pending.push_back(std::move(p));

            if ((int64_t)pending.size() >= infer_batch_size) {
                evaluate_pending(pending, add_noise, eval_fn, noise_fn);
                pending.clear();
            }
        }
        if (!pending.empty()) {
            evaluate_pending(pending, add_noise, eval_fn, noise_fn);
            pending.clear();
        }

        return get_results(0);
    }

    // 落子后返回对应子节点 id（供下一手树复用），未创建返回 -1
    int64_t next_root(int64_t root_id, int64_t flat_move) {
        if (root_id < 0 || root_id >= next_id)
            throw std::runtime_error("root_id 越界");
        return children[(size_t)root_id * kMoves + flat_move];
    }

    // ---- 调试用：导出当前树状态（供与 Python 后端逐位比对）----
    py::dict dump_state() {
        py::ssize_t n = (py::ssize_t)next_id;
        py::array_t<int32_t> children_a({n, (py::ssize_t)kMoves});
        std::memcpy(children_a.mutable_data(), children.data(),
                    (size_t)n * kMoves * sizeof(int32_t));
        py::array_t<double> priors_a({n, (py::ssize_t)kMoves});
        std::memcpy(priors_a.mutable_data(), priors.data(),
                    (size_t)n * kMoves * sizeof(double));
        py::array_t<double> clean_a({n, (py::ssize_t)kMoves});
        std::memcpy(clean_a.mutable_data(), clean_priors.data(),
                    (size_t)n * kMoves * sizeof(double));
        py::array_t<int32_t> visits_a(n);
        std::memcpy(visits_a.mutable_data(), visits.data(),
                    (size_t)n * sizeof(int32_t));
        py::array_t<int32_t> virt_a(n);
        std::memcpy(virt_a.mutable_data(), virt.data(),
                    (size_t)n * sizeof(int32_t));
        py::array_t<double> w_sum_a(n);
        std::memcpy(w_sum_a.mutable_data(), w_sum.data(),
                    (size_t)n * sizeof(double));
        py::array_t<double> val_a(n);
        std::memcpy(val_a.mutable_data(), val.data(),
                    (size_t)n * sizeof(double));
        py::array_t<double> value_a(n);
        std::memcpy(value_a.mutable_data(), value.data(),
                    (size_t)n * sizeof(double));
        py::array_t<bool> exp_a(n);
        {
            bool* d = exp_a.mutable_data();
            for (py::ssize_t i = 0; i < n; ++i) d[i] = expanded[(size_t)i];
        }
        py::array_t<bool> tf_a(n);
        {
            bool* d = tf_a.mutable_data();
            for (py::ssize_t i = 0; i < n; ++i) d[i] = term_flag[(size_t)i];
        }
        py::array_t<double> tv_a(n);
        std::memcpy(tv_a.mutable_data(), term_val.data(),
                    (size_t)n * sizeof(double));
        py::array_t<int32_t> parent_a(n);
        std::memcpy(parent_a.mutable_data(), parent.data(),
                    (size_t)n * sizeof(int32_t));
        py::array_t<int32_t> move_a(n);
        std::memcpy(move_a.mutable_data(), move_flat.data(),
                    (size_t)n * sizeof(int32_t));

        return py::dict(
            py::arg("children") = children_a, py::arg("priors") = priors_a,
            py::arg("clean_priors") = clean_a, py::arg("visits") = visits_a,
            py::arg("virt") = virt_a, py::arg("w_sum") = w_sum_a,
            py::arg("val") = val_a, py::arg("value") = value_a,
            py::arg("expanded") = exp_a, py::arg("term_flag") = tf_a,
            py::arg("term_val") = tv_a, py::arg("parent") = parent_a,
            py::arg("move_flat") = move_a, py::arg("next_id") = py::int_(next_id));
    }
};

PYBIND11_MODULE(mcts_core, m) {
    m.doc() = "五子棋 MCTS C++ 核心：树操作与统计全部在 C++ 内完成，"
              "网络推理与 Dirichlet 噪声由 Python 回调提供。";
    py::class_<MctsTree>(m, "MctsTree")
        .def(py::init<double, double, double>(), py::arg("c_puct") = 1.0,
             py::arg("dirichlet_alpha") = 0.1, py::arg("noise_eps") = 0.25)
        .def_readwrite("c_puct", &MctsTree::c_puct)
        .def_readwrite("dirichlet_alpha", &MctsTree::dirichlet_alpha)
        .def_readwrite("noise_eps", &MctsTree::noise_eps)
        .def("run", &MctsTree::run, py::arg("board"),
             py::arg("num_simulations"), py::arg("cur_root") = py::none(),
             py::arg("add_noise") = false, py::arg("infer_batch_size") = 64,
             py::arg("eval_fn"), py::arg("noise_fn"),
             "执行一次搜索，返回 (root_value, probs[15,15], root_id)。")
        .def("next_root", &MctsTree::next_root, py::arg("root_id"),
             py::arg("flat_move"),
             "返回落子对应的子节点 id，未创建返回 -1。")
        .def("dump_state", &MctsTree::dump_state,
             "导出当前树状态（截至 next_id），供与 Python 后端比对。")
        .def("node_count", [](const MctsTree& t) { return t.next_id; });
}
