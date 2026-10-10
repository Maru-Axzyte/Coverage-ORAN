# Chạy mô hình UAV: GA + WE top-K + CMA + MILP

Mở **`main.py`** trong VS Code và bấm **Run Python File**.
Hoặc nhấn F5, chọn **Main: WE + CMA + MILP**. Cấu hình F5 dùng `.venv` và không
ghi đè số thế hệ hay thông số môi trường bằng các argument ẩn.

```powershell
.\.venv\Scripts\python.exe main.py
```

Mặc định chương trình chạy mô hình, lưu PNG/JSON cùng thư mục mã nguồn rồi mở
hình placement, service/resources và chất lượng surrogate. Timeline theo kỳ
WE được lưu với hậu tố `_all_steps.png`. Tên kết quả giữ tiền tố
`note_we_2d_convergence` như trước để không làm hỏng các công cụ đọc kết quả.

Nếu chỉ muốn lưu hình mà không mở cửa sổ:

```powershell
.\.venv\Scripts\python.exe main.py --no-show --no-open
```

## File nào phụ trách việc gì?

| File | Trách nhiệm |
|---|---|
| `main.py` | Điểm chạy duy nhất: đọc cấu hình, chạy optimizer, lưu kết quả và gọi vẽ |
| `config.py` | Thông số môi trường, vô tuyến, tài nguyên, solver |
| `env.py` | Môi trường, UE/UAV, hình học và các công thức vật lý |
| `scenario.py` | Tạo kịch bản từ `config.py` bằng các hàm trong `env.py` |
| `milp.py` | Bài toán MILP phân bổ UE, PRB và công suất |
| `evaluation.py` | Nối placement với MILP, cache và đếm số lần giải |
| `ga.py` | `GASettings`, cá thể/archive, chọn bố mẹ, SBX, mutation và hàm đánh giá dùng chung |
| `contextual_ga.py` | Gắn học trọng số vào GA, context, sinh con độc lập, lịch số cá thể |
| `preference_weights.py` | Các bài toán học trọng số bằng CVXPY |
| `contextual_weight_audit.py` | Chuẩn bị/kiểm tra bằng chứng để học B*c |
| `surrogate.py` | Dự đoán transition, covariance, residual, snapshot và thay nhãn exact |
| `cma.py` | Sinh mẫu CMA và cập nhật mean, p, C, sigma |
| `joint_topk_we.py` | Xếp hạng top-K, elite/boundary/audit và vòng lặp WE + CMA chung |
| `optimizer.py` | Lắp các thành phần thành một optimizer, truyền cấu hình đúng một lần |
| `plots.py` | Toàn bộ code vẽ bản đồ 2D, timeline, service/resources, chất lượng surrogate |
| `test_settings_cleanup.py` | Kiểm tra cấu hình, nối module và vòng WE nhỏ bằng allocator giả |
| `test_weight_learning.py` | Kiểm tra bộ học weights và quan hệ với comparator |
| `test_shared_prb_pool.py` | Kiểm tra quỹ PRB chung bằng các MILP rất nhỏ |
| `test_plot_quality.py` | Kiểm tra thống kê chất lượng theo generation |

Luồng gọi:

```text
main.py
  ├─ scenario.py → env.py + config.py
  ├─ evaluation.py → milp.py → env.py
  ├─ optimizer.py
  │    └─ joint_topk_we.py
  │         ├─ ga.py + contextual_ga.py
  │         ├─ surrogate.py
  │         ├─ cma.py
  │         └─ evaluation → MILP
  └─ plots.py → PNG; main.py → JSON
```

Các module thuật toán không import `main.py` hoặc `plots.py`; không có mô phỏng
tự chạy khi import. Các biểu đồ nhận `PlotSettings` riêng, nên chế độ `--quick`
không sửa biến toàn cục rồi làm lệch nhãn/số lượng UAV của lần chạy tiếp theo.

## Chỉnh thông số

- Môi trường: sửa `config.py`.
- GA/WE: sửa `GASettings` trong `ga.py` (đặc biệt `population_size`,
  `population_capacity`, `generations`).
- `main.py --help` liệt kê các tuỳ chọn ghi đè có chủ ý cho một lần chạy.
- `optimizer.py` không còn ghi đè ngầm block SBX, số con/cặp hay bật ràng buộc
  khoảng cách. Các lựa chọn đó nằm trong `GASettings`; khoảng cách mặc định
  lấy từ `config.uav_min_separation_m`.

Các giá trị được giữ khi dọn mã: 200 cá thể đầu, trần 5000, 400 thế hệ,
40 mẫu MILP khởi tạo và tối đa 50 con/cặp (bị giới hạn bởi chỗ còn trống).
Đây chỉ là mô tả; giá trị thực lấy từ cấu hình hiện tại, không lấy từ README.

| Nhóm còn dùng trong `GASettings` | Tham số |
|---|---|
| Kích thước và thời gian | `population_size`, `population_capacity`, `generations` |
| MILP khởi tạo và học weights | `initial_milp_seeds`, `update_period` |
| Sinh con | `sbx_eta`, `sbx_uav_blocks`, `children_per_pair`, `mutation_probability`, `mutation_sigma` |
| Giữ tinh hoa | `elite_fraction` |
| Trọng số khởi tạo | `beta_power`, `mu_prb`, `eta_bottleneck` |
| Dự đoán transition | `transition_state_bandwidth`, `transition_action_bandwidth`, `transition_neighbors` |
| Khoảng cách và tái lập | `include_separation_constraint`, `min_uav_separation_m`, `random_seed` |

`population_size` là số cấu hình ban đầu, không phải số UAV. Mỗi cấu hình
chứa đủ các UAV. `population_capacity` là trần cùng lúc của cha mẹ và con;
lịch số cấu hình sống sót nằm trong `ContextualCSAEA._population_target`.
Lịch này, ngân sách MILP, top-K và CMA không thay đổi trong đợt dọn mã.

Đã bỏ các nút chỉnh không tác động đến luồng joint đang chạy:
`clusters`, `light_violation_epsilon`, `light_violation_fraction`,
`bottleneck_target`, `idle_uav_redeployment_weight`, `unserved_infill_variants`,
`surrogate_service_tie_epsilon`, `omega_uncertainty`, `omega_overlap`,
`jaccard_weight`, `radio_similarity_weight`, `topology_similarity_weight`,
`transition_risk_kappa`, `use_war_elimination`.

Chỉ còn bộ dự đoán trong `surrogate.py`, dùng `joint_records`; đã bỏ bộ
`TransitionRecord`/`self.transitions` trùng ở `ga.py` và descriptor chỉ phục vụ
bộ cũ. Metadata `joint_transition_window_size` đếm cửa sổ transition đang dùng,
không phải tổng số lần MILP. Các khóa JSON lịch sử rỗng như `war_history` được
giữ để tương thích công cụ đọc kết quả, không có vòng chiến tranh cũ chạy ngầm.

Không tự ý kích hoạt lý thuyết từng được đề xuất nhưng chưa áp dụng: bottleneck
hiện là `max(peak_power_load, system_prb_load)`, không trừ `bottleneck_target`;
idle-UAV là chỉ số chẩn đoán, không có penalty; comparator hiện so trực tiếp
`S`, không dùng `surrogate_service_tie_epsilon`. Deb, epsilon theo thế hệ,
`f_resource` và bộ học contextual weights được giữ nguyên.

PRB hiện dùng quỹ chung 135 cho toàn hệ, không chia cứng 27/UAV. Một UAV có
thể dùng 40 PRB nếu tổng các UAV không vượt 135. `uav_max_prb = 135` là cận
trên riêng suy ra từ quỹ chung, không phải mỗi UAV được cấp thêm 135 PRB.
MILP vẫn giữ ràng buộc tổng; surrogate/WE dùng tải PRB `sum(N_u)/135`.
Giới hạn 2 W/UAV và 10 PRB/link không đổi. Kết quả đã lưu trước thay đổi vẫn
thuộc cấu hình cũ; chạy lại `main.py` để tạo kết quả theo quỹ chung.

## Nhánh cũ đã bỏ

Đã bỏ `war_scheduler.py`, các hàm matching/layered survival và các vòng lặp WE
cũ. Các runner `run_hybrid_we_cma_demo.py`, `run_matching_we_demo.py`,
`run_note_we_demo.py` đã được thay bằng `main.py`; `joint_topk_plots.py` được
gộp vào `plots.py`. Dùng main từ nay; các argument `--war-method`, `--student-t`
và so sánh baseline cũ không còn thuộc chương trình này.

Bản sao trước khi tách ở `tmp/module_split_before_20261004_155136.zip`, gồm mã
Python và cấu hình VS Code trước thay đổi. Các kết quả mô phỏng cũ không bị xóa.

## Kiểm tra

Theo yêu cầu rút gọn, mã dự án không còn các lệnh `raise ValueError` tự viết
để kiểm tra đầu vào. Các hàm nội bộ giả định cấu hình hợp lệ: số lượng/ngân sách
dương, ma trận đúng kích thước, bandwidth dương và lựa chọn mode hợp lệ.
Điều này không làm mất các ràng buộc tối ưu của MILP hay thay công thức vật lý.
`main.py` vẫn kiểm tra các tùy chọn dòng lệnh; lỗi từ Python/thư viện vẫn có thể
xuất hiện. Kết quả học weights không hợp lệ sẽ không được cập nhật, và CMA
không cập nhật từ mẫu chưa được MILP xác nhận. Không có cơ chế bỏ qua mọi lỗi.

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_settings_cleanup test_weight_learning test_shared_prb_pool test_plot_quality -v
```

Kiểm thử nối luồng dùng allocator giả với hai UAV và sáu thế hệ; các kiểm thử
PRB chỉ giải MILP tổng hợp rất nhỏ. Đây là kiểm tra hồi quy, không phải chứng
minh hiệu quả tối ưu trên thí nghiệm thật. Chạy `main.py` để tự làm thí nghiệm.
