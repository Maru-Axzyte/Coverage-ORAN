# Biểu đồ cột trung bình kết quả đã lưu

Hình hiện hiển thị WE, Zhong GA–MILP và Random Search–MILP. PSO–MILP đã được ẩn theo yêu cầu; dữ liệu PSO trong snapshot/CSV vẫn được giữ. Nhãn trong cột chỉ có giá trị, không có `n=...`.

Chạy từ thư mục dự án:

```powershell
& "C:\Users\Hoa Ly\AppData\Local\Programs\Python\Python313\python.exe" -B comparison_all_algorithms/mean_metric_bars.py
```

Chỉ đọc kết quả có sẵn và vẽ, không gọi MILP hay chạy optimizer. Zamani không được đọc hoặc đưa vào bộ hình này. GA thường chưa hoàn thành không được biểu diễn bằng cột 0; đường GA có trong hình là Zhong GA–MILP thích nghi.

- `mean_metrics_overview.png` / `.svg`: hình tổng hợp bốn tiêu chí.
- `mean_coverage_bars`, `mean_power_bars`, `mean_prbs_bars`, `mean_fitness_bars`: hình riêng từng tiêu chí, có PNG và SVG.
- `summary.csv` / `.json`: trung bình, số map và thông tin dữ liệu.
- `source_snapshot.json`: dữ liệu rút gọn của từng run, danh sách file nguồn, SHA256, file bị loại và file chưa hoàn thành. Dùng `--from-cache` để vẽ lại cùng snapshot, không đọc lại các thí nghiệm đang chạy.

Mỗi cột là trung bình số học của nghiệm cuối trên toàn bộ map hoàn thành, được chấp nhận của chính phương pháp đó. Đây là thống kê mô tả trên các tập map có thể khác nhau, không phải trung bình ghép cặp trên cùng map. Các nguồn trùng nhau được loại trùng; xung đột giữa các bản lặp được ghi nhận và loại thay vì chọn bản tốt hơn. Nhãn trong cột chỉ hiển thị giá trị; số map vẫn được giữ trong CSV/JSON, không hiển thị trên hình. Riêng WE trong ô fitness hiển thị hai đầu khoảng giá trị xác định được.

Coverage là số UE được phục vụ đủ demand chia số UE của map, tính theo phần trăm. Run hoàn thành nhưng không có nghiệm cuối khả thi có coverage bằng 0; tài nguyên và fitness không xác định bị để thiếu, không gán 0. Power là tổng RF power (W), PRB là tổng toàn hệ thống. Cột power/PRB có thể thấp hơn khi phục vụ ít UE hơn, nên phải đọc cùng coverage.

Ô fitness là **điểm tham chiếu chấm lại**, không phải fitness theo trọng số học được riêng của từng run:

`f_ref = -(P_total/10 + N_total/135 + z)/3`,

`z = max(max_u(P_u/2), N_total/135)`.

Trọng số cố định `(1/3,1/3,1/3)` được dùng cho mọi phương pháp nhằm có cùng thước đo. Đây chỉ là hậu xử lý; không đổi mục tiêu hoặc kết quả tối ưu của các phương pháp đã chạy. f_ref chỉ xét tài nguyên, không thay thế coverage và không chứa ưu tiên admission.

PSO, Zhong và Random có công suất từng UAV nên tính được điểm tham chiếu chính xác cho allocation đã lưu. WE thường chỉ có tổng công suất. Với nghiệm khả thi:

`max(P_total/10, N_total/135) <= z <= max(min(1, P_total/2), N_total/135)`.

Tính f_ref ở hai đầu rồi lấy trung bình hai đầu qua các map cho khoảng hợp lệ của trung bình WE. Vùng gạch chéo là **khoảng xác định do thiếu dữ liệu**, không phải khoảng tin cậy thống kê. Không lấy trung điểm làm fitness. Khi dùng đủ 135 PRB hoặc các cận trùng nhau, có thể xác định chính xác điểm tham chiếu dù thiếu công suất từng UAV. Cột WE phần tô nhạt chỉ đi tới một đầu khoảng; endpoint thật nằm trong dải gạch chéo.

Không có phép giải lại allocation để đoán fitness gốc. Đọc `mean_f_ref_lower` và `mean_f_ref_upper` trong CSV nếu cần giá trị đầy đủ.
