# Zhong-inspired GA–MILP

Trong thư mục này, chạy `python main.py` (hoặc mở main.py rồi Run Python File).
Đường dẫn mặc định độc lập với thư mục terminal hiện tại.

Chương trình chọn ngẫu nhiên có phân tầng 20 map trong bộ 300 kết quả WE đã lưu.
Seed chọn map mặc định 20261013, cùng cách chọn với đối chứng PSO trước đó.
Danh sách và dữ liệu UE/demand được cố định trong `results/selection.json` và
`results/inputs/`. Không sinh lại UE. Mỗi map dùng ngân sách MILP của kết quả WE
tương ứng, bao gồm khởi tạo; cache hit không tiêu thêm ngân sách.

Không thay đổi file hay thông số thuật toán chính. Chỉ import môi trường,
scenario và MILP chung. Kiểm tra fingerprint ngăn so sánh nhầm với mô hình vật lý
đã thay đổi so với thí nghiệm WE lưu trước đó.

## Thuật toán

Tham khảo: https://arxiv.org/abs/2008.03125 (Algorithm 1, Section IV).
Đây là bản thích nghi, không phải tái hiện nguyên bản bài báo.

- Gene mỗi UAV: (x,y,h,theta), thay cho (x,y,R); không dùng bước suy độ cao gốc.
- Khởi tạo bằng tâm đường tròn qua ba UE; thử lại khi suy biến/ngoài miền/vi phạm
  khoảng cách. Sau 100 lần thử cho một UAV chuyển sang vị trí ngẫu nhiên trong miền.
- Độ cao/góc liên tục, ngẫu nhiên trong giới hạn config.
- Chọn một đối thủ ngẫu nhiên mỗi thế hệ, sao chép nó thay các cá thể điểm thấp hơn.
  Đây là cách diễn giải đoạn mô tả bằng lời; không phải roulette wheel chuẩn.
- Lai ghép tại giữa các khối UAV, đột biến thay ngẫu nhiên một khối.
- Thao tác vi phạm hình học bị hoàn tác; không nới khoảng cách. Khởi tạo thất bại
  sau giới hạn số lần thử thì dừng map và báo lỗi.
- Fitness là số UE phục vụ của cấu hình khả thi. Không Deb, weights, WE, CMA, surrogate.
- MILP giữ nguyên mục tiêu phân bổ hiện có. Cần solver_optimal để chấp nhận nhãn.
- Archive báo cáo giữ nghiệm khả thi có số UE cao nhất; hòa thì giữ nghiệm gặp trước.
  Archive không được chèn trở lại quần thể, không thêm elitism.
- Mặc định population=100, pc=0.8, pm=0.01, tối đa 17000 vòng (tham số tham khảo
  bài báo, không tuyên bố tối ưu cho môi trường này). Thường dừng trước theo ngân sách MILP.
- Cache giữ cấu hình trùng, không tự thêm random restart. Nếu chạm giới hạn thế hệ
  trước ngân sách, kết quả ghi rõ `generation_limit_before_budget`.

## Chạy

```powershell
python main.py
```

Các lựa chọn:

```powershell
python main.py --prepare-only
python main.py --plot-only
python main.py --output-dir results_new --selection-seed 20261014
```

`--prepare-only` chỉ đọc/lưu danh sách map, không giải MILP. Có thể cập nhật fingerprint
runner của một kế hoạch chưa có kết quả bằng lệnh này, nếu nguồn và danh sách không đổi.

Log MILP in từng cấu hình; tổng kết GA mỗi 10 thế hệ và khi hoàn tất map. Dòng
`solving...` có nghĩa đang giải một cấu hình, không nhất thiết chương trình bị treo.

Chạy lại cùng lệnh sẽ bỏ qua các map hoàn tất. Map dở chạy lại từ đầu với cùng seed,
không tiếp tục từ thế hệ đang dở; log nối thêm một lượt chạy. Không chạy hai phiên
cùng output-dir đồng thời. Đổi tham số/code sau khi có kết quả: dùng output-dir mới.

## Đầu ra

- `results/logs/ue_XXX_map_YYY.log`: log theo map.
- `results/runs/ue_XXX_map_YYY.json`: thông số, từng đánh giá MILP, history và nghiệm tốt nhất.
- `results/comparison.csv`, `comparison.json`: các cặp kết quả WE/GA đã hoàn tất.
- `results/comparison.png`: số UE phục vụ trung bình theo nhóm số UE; cập nhật sau mỗi map.
- Terminal in số UE trung bình của hai phương pháp trên các map đã hoàn tất.

Biểu đồ không bật cửa sổ chặn lượt chạy tiếp theo; mở comparison.png để xem.
WE được đọc từ kết quả cũ, không chạy lại. Không dùng nghiệm WE để khởi tạo GA.
Các map dừng lỗi chưa được tính là map hoàn tất; xem số lượng completed và log.

## Giới hạn kết luận

Mục tiêu đối chứng chính: số UE đáp ứng QoS và khả thi, theo cùng ngân sách.
Đây là so sánh hai hệ thống hoàn chỉnh, không cô lập riêng tác dụng của WE.
Khởi tạo khác nhau là một phần của mỗi thuật toán; không phải ablation khởi tạo chung.
GA không có mục tiêu tài nguyên ngoài cùng. Các số power/PRB chỉ mang tính mô tả;
cột resource_comparable đòi hỏi cùng số UE và tổng demand, cả hai khả thi.
Đó vẫn không phải cùng tập UE, không chứng minh tối ưu năng lượng.
Thời gian WE lịch sử có thể từ máy khác, không dùng để kết luận nhanh hơn.
20 map, một seed thuật toán/map là thử nghiệm ban đầu, không chứng minh vượt trội tổng quát.

## Kiểm tra không chạy mô phỏng

```powershell
python -B -m unittest discover -s . -p test_baseline.py
```

Các test dùng bộ đánh giá giả lập, không gọi MILP thật.
