# WE theo note: thay đổi ngày 29/09/2026

> Tài liệu lịch sử của nhánh matching đã bỏ. Hiện tại chạy `main.py`, dùng
> joint top-K theo phần note cập nhật trang 11--14. Xem `README.md` và
> [JOINT_TOPK_RESTORATION.md](JOINT_TOPK_RESTORATION.md) cho luồng hiện tại.

Đối chiếu `Downloads/私のノット.pdf` (15 trang, bản sửa 28/09/2026), tập trung
phần SBX trang 5–6, WE trang 6–9 và transition trang 9–12. Đây là sửa cơ chế WE
và truyền gene, không phải tuyên bố mọi công thức trong note đã được triển khai.

## Chạy và xem hình

Hướng dẫn hiện tại: mở `main.py`, chọn **Run Python File**, hoặc nhấn F5 chọn
**Main: WE + CMA + MILP**. Các cơ chế matching dưới đây là ghi chép lịch sử,
không còn được chương trình gọi.

Hình và JSON mới có tiền tố `note_we_2d_convergence`, không ghi đè kết quả cũ.
Các runner Student-t/matching cũ không còn là điểm chạy của chương trình.

Cập nhật môi trường theo bộ thông số người dùng gửi: `config.py` là nguồn chuẩn.
5 UAV, 50 UE, 1000 × 1000 m, 135 PRB trực giao, 27 PRB/UAV, 2 W/UAV;
mỗi link tối đa 10 PRB và 1 W, bước công suất 0.1 W. Radio: 3.5 GHz,
NF=9 dB, 360 kHz/PRB, h trong [250,300] m, theta trong [30,60] độ.
Thông số hiện tại lấy từ `GASettings` và `config.py`; xem `README.md`.

## Một cá thể là gì?

Một cá thể chứa TOÀN BỘ U UAV, ma trận U × 4 với hàng u là (x,y,h,theta).
Các con là những cấu hình thay thế, không phải UAV mới đang cùng hoạt động.
P, PRB, association, V,C,S và fitness không phải gene được sao chép từ cha mẹ.

## Truyền gene

- Chọn cha mẹ bằng tournament. Ưu tiên hai cấu hình khác nhau; nếu quần thể đã
  trùng hoàn toàn, cho phép fallback và dùng mutation để khôi phục đa dạng.
- Trong chế độ note, mỗi UAV dùng một hệ số SBX chung cho bốn gene. Trước mutation,
  hai con đối xứng qua trung điểm cha mẹ; đây là **biến thể SBX theo khối** từ công
  thức vector trong note, không phải SBX từng tọa độ cũ.
- Ghép theo đúng hàng/ID UAV. Không tự hoán đổi nhãn UAV giữa hai cha mẹ.
- Mỗi cặp tạo tối đa 10 con độc lập từ hai cha mẹ ban đầu; cặp cuối có thể ít hơn
  vì ngân sách sinh sản/trần quần thể. Không dùng con trước làm mẹ của con sau.
- Số 10 là lựa chọn vận hành, không phải số đã chứng minh tối ưu. Ngân sách tổng
  sinh con mỗi thế hệ không tăng. `GASettings.children_per_pair` điều chỉnh nó.
- Mutation giữ nguyên. Một tổ hợp tốt vẫn có thể bị phá vỡ; luôn phải đánh giá
  lại cấu hình con. Không hứa con tốt hơn mẹ hoặc hội tụ toàn cục.
- Ghi lại anchor đã xác minh được dùng khi dự đoán. Nếu sau đó MILP đánh giá
  con, transition có đúng trạng thái trước và sau, không làm mất record.

## WE: before / during / after

1. **Before:** gộp cha mẹ và tất cả con, bỏ bản trùng, ưu tiên bản đã xác minh;
   cố định epsilon, weights và dữ liệu sai số cho lượt này. Giữ incumbent exact.
2. **Warm-up gia đình:** tất cả cá thể duy nhất được đấu hoặc nghỉ; cha mẹ dùng
   chung chỉ xuất hiện một lần. Không loại ai ở vòng đầu. Các cha mẹ chưa thuộc
   nhóm nào vẫn tham gia nhóm còn lại.
3. **Toàn cục:** tối đa ba vòng, không ép gia đình nào phải có người sống sót.
   Xóa điểm vòng gia đình vì các gia đình có độ khó đối thủ khác nhau.
4. Điểm s: thắng 1, hòa 0.5, nghỉ 0. Buchholz B là tổng điểm các đối thủ đã gặp;
   s và B là hai biến khác nhau. Ghép trận tối thiểu tổng |s_i-s_j| bằng bộ giải
   chuyên biệt có sẵn. Nếu thêm cấm tái đấu thì không còn dùng nguyên bộ giải này.
5. Thắng/thua vẫn do (V,C,S,f_resource), không do s hoặc Buchholz. Điểm và Buchholz
   chỉ phá hòa khi khóa chất lượng bằng nhau; không được đánh đổi service.
6. Chỉ những ứng viên ở phần đuôi ngoài sức chứa mới có thể bị loại. Thua một
   đối thủ mạnh không đủ để loại một cá thể đang nằm trong nhóm tốt cần giữ.
   Ở vòng giữa, xét người thua ít nhất hai trận hoặc bị đánh dấu drop tạm.
7. Hạn mức loại lũy kế là ceil(D/3), ceil(2D/3), D, với D là số ứng viên dư.
   Đây là lịch đơn giản có giới hạn, không phải tỷ lệ tối ưu từ lý thuyết.
   Cuối vòng cuối phải đóng tập dự bị và cắt theo sức chứa/comparator nếu cần.
8. **After:** chỉ người sống sót được quay lại sinh sản. Người đã loại không được
   ghép ngược từ danh sách gốc để lấp chỗ. Không thêm ứng viên giả cho đủ target.

Bốn tầng elite/high/medium/low được ghi theo bốn phần của thứ hạng hiện tại.
Đây là nhãn tiềm năng tương đối để kiểm tra log, KHÔNG phải bốn quần thể độc lập,
không có hạn ngạch giữ bắt buộc từng tầng. Tỷ lệ loại tăng theo vòng toàn cục.
Không thêm migration/repatriation, thưởng điểm nghỉ hoặc bộ luật cờ vua đầy đủ.

## Dự bị và kiểm tra MILP

Sửa lỗi cấu trúc cũ: không yêu cầu đủ K cá thể đã MILP mới có ngưỡng cạnh tranh.
Lấy cấu hình thứ K trong bảng xếp hạng tạm làm tham chiếu. Nếu tham chiếu cũng
chỉ là dự đoán thì dùng cả khoảng [J_cut_minus,J_cut_plus] của nó:

- keep tạm nếu J_i_plus < J_cut_minus;
- drop tạm nếu J_i_minus > J_cut_plus;
- còn lại reserve.

Chỉ áp dụng khi V,C,S tương đương theo khóa so sánh. Nếu không tương đương,
Student-t không chứng nhận được ưu tiên service: để reserve và dùng comparator
chính cho quyết định sức chứa. Chế độ strict-priorities vẫn từ chối coi các ưu
tiên dự đoán là đã xác minh. Pool chưa vượt target thì chưa cần competitive cut.

Một ngân sách MILP dùng chung cho mọi gia đình/vòng (mặc định runner cấp 1/lượt
thế hệ chính). Ưu tiên dự bị có trận thua gần ranh giới giữ/loại. Audit định kỳ dùng
ngay ngân sách đó, không cộng một lần giải cho từng cặp. Sau MILP, tính lại kết quả
giao tranh trước khi loại; replay điểm cũ nếu một nhãn dự đoán đã bị sửa.

Giữ nguyên infill định kỳ và CMA ngoài ngân sách trên. Không phải toàn chương
trình chỉ gọi MILP một lần mỗi thế hệ.

Log `removal_reason` phân biệt `exact_quality`, `nominal_interval` và
`capacity_prediction`. Keep/drop là tạm thời, không phải đảm bảo đúng 95%.
Nhiều ứng viên vẫn có thể vào reserve khi C,S không chắc hoặc thiếu mẫu;
không thay tên nhóm để làm đồ thị trông đẹp hơn.

## Giữ nguyên và chưa giải quyết trong lần sửa này

- Lần sửa WE không đổi B*c/CMA. Lần cập nhật cấu hình sau đó đã đổi mode công
  suất sang bước W và áp dụng quota PRB cùng giới hạn mỗi link theo yêu cầu.
- Giữ J ba thành phần của nhánh Student-t, chưa đưa lại idle penalty trong note.
- Giữ comparator transitive theo khóa số học; không chép dung sai theo từng cặp
  S trong note vì nó có thể gây vòng so sánh không bắc cầu.
- Note ghi beta>1 là con giống mẹ: điều kiện đúng là beta=1. Eta cao làm phân bố
  beta tập trung gần 1, không phải luôn làm beta tăng.
- Chưa thêm trust-region, mô hình độ tin cậy V/C/S hay cơ chế số con học tự động.
- Student-t với dữ liệu online không có bảo đảm iid; intervals dùng để sàng lọc
  danh nghĩa, không chứng minh giữ đúng top K hay phủ tối đa UE.
- Trọng số B*c vẫn có giới hạn học từ các cặp Pareto; sửa WE không giải quyết
  khả năng nhận diện sở thích trade-off của bộ học weights.
- Phần CMA hiện vẫn có prescreening rồi exact subset, không phải mọi mẫu sinh
  ra đều gọi MILP như một phiên bản lý thuyết ở trang 13.

## Kiểm tra

Unit test dùng allocator GIẢ, không giải HiGHS/CVXPY và không chạy mô phỏng UAV:

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_joint_topk_we -v
```

Trong lần sửa lịch sử trước khi đổi sang joint top-K, đã qua 39 test, gồm độc lập giữa anh chị em, vector SBX, lưu anchor,
không đếm trùng, giữ tinh hoa, cứu người thua sau kiểm chứng, ngân sách chung,
ghép trận tối ưu trên các trường hợp nhỏ, giới hạn quần thể và đóng tập dự bị.
Đây là kiểm tra tính đúng của cơ chế, không phải kiểm chứng tăng số UE phục vụ.
Chưa chạy bài toán nghiên cứu đầy đủ 300 thế hệ sau khi đồng nhất cấu hình.
