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
| `test_joint_topk_we.py` | Kiểm thử nhỏ bằng allocator giả, không chạy MILP nghiên cứu |

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
- Cách lắp cấu hình trong `optimizer.py` giữ nguyên mode đang dùng: block SBX,
  10 con/cặp và ràng buộc khoảng cách theo config. Không thay các trị số đó khi
  tách module.

Ở thời điểm tách: 200 cá thể đầu, trần 1000, 300 thế hệ; môi trường 5 UAV/50 UE,
1000x1000 m, 135 PRB và 2 W/UAV. Đây là mô tả, không phải cấu hình thứ hai.

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
`NOTE_WE.md` là ghi chép lịch sử; mô tả lý thuyết hiện tại nằm trong
`JOINT_TOPK_RESTORATION.md`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_joint_topk_we -v
```

Kiểm thử nối toàn luồng dùng allocator giả, lưu đủ PNG/JSON trong thư mục tạm.
Một ca 6 thế hệ cùng seed đã được so sánh trước/sau tách file: lịch sử WE,
top-K, audit và CMA không đổi. Điều này kiểm tra việc tách module, không chứng
minh hiệu quả tối ưu trên thí nghiệm thật. Không chạy lại mô phỏng nghiên cứu
trong lần tổ chức mã nguồn này.
