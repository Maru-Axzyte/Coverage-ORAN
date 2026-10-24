# Khảo sát đối chứng cho placement UAV và phân bổ tài nguyên bằng MILP

Ngày khảo sát: 16/10/2026.

## Kết quả lựa chọn

Thứ tự triển khai ban đầu là **DE–MILP**, **GPSAF–DE–MILP** và **Local Branching trên tập vị trí ứng viên**, xét cả công sức tích hợp. Đánh giá sâu hơn dưới đây phân biệt thứ tự dễ triển khai với kỳ vọng cải thiện. Các phương pháp mới chưa được benchmark trên dự án.

Khảo sát có chọn lọc 16 công trình, sử dụng tóm tắt, các phần phương pháp truy cập được và tài liệu/mã của tác giả hoặc bên triển khai. Không phải systematic review; không tuyên bố đã tái hiện 16 bài. Các nhận định về khả năng ghép vào dự án là suy luận từ mô hình và mã hiện tại. Những bài chưa có mã được xác minh không bị coi là không có mã.

## Đánh giá lại theo khả năng cải thiện

Đọc lại phần phương pháp, thí nghiệm và giới hạn của các ứng viên chính dẫn tới các lựa chọn theo mục tiêu sau. Đây là thứ tự ưu tiên thử nghiệm có điều kiện, không phải xếp hạng thực nghiệm hoặc xác suất thắng đã đo.

| Mục tiêu | Lựa chọn ưu tiên | Căn cứ và giới hạn |
|---|---|---|
| Cải thiện một nghiệm tốt đã có của WE hoặc PSO | Restricted-neighborhood MILP lấy cảm hứng Local Branching | Khai thác trực tiếp cấu trúc phân bổ, cho một/hai UAV đổi vị trí trong tập nhỏ, reoptimize association/resources của toàn hệ. Cần giữ chính xác incumbent trong tập ứng viên và giữ nghiệm cũ khi không cải thiện. Đây là phần refinement bổ sung, không phải một baseline độc lập; phải tính ngân sách seed. |
| Một phương pháp mới chạy độc lập, overhead nhỏ | L-SHADE–MILP thích nghi | Có success-history, current-to-pbest và archive; phù hợp thử nghiệm tìm kiếm 20 chiều không cần surrogate. Bằng chứng gốc dùng ngân sách rất khác; độ tin cậy về ưu thế trên dự án chỉ ở mức vừa. Cần tuning population riêng và xử lý đúng feasibility/fitness improvement. |
| Giảm số đánh giá thật / đối chứng surrogate | GPSAF–PSO–MILP | Trong Bảng III của GPSAF, average rank GPSAF–PSO là 3.214, GPSAF–DE là 3.857, GPSAF–GA là 3.964; nhỏ hơn tốt hơn. GPSAF–ISRES còn tốt hơn (3.036), SACOBRA tốt nhất bảng (2.321). Vì thế không gọi GPSAF–PSO là tốt nhất nói chung. PSO là base có bằng chứng thuận lợi trên log của dự án; chọn GPSAF–PSO là suy luận kết hợp, chưa được đo. |
| Lựa chọn có bằng chứng trực tiếp trên dự án | PSO–MILP hiện có | Snapshot tổng hợp ngày 15/10: 34 map ghép cặp, WE thắng số UE ở 1, hòa 22, thua 11. Đây là đối chứng thực nghiệm quan trọng; không quy kết kết quả này cho nguyên bản nhánh PSO của Zamani. |

### Những giới hạn bằng chứng cần tính vào lựa chọn

- [L-SHADE, Mục III](https://ryojitanabe.github.io/pdf/tf-cec2014.pdf) dùng 30 bài CEC2014, 10/30/50/100 chiều, 51 run, ngân sách `10000*D`; không phải 1280 calls. Cấu hình tuned `N_init=18D` sẽ là 360 cá thể ở 20 chiều, tiêu khoảng 28% của 1280 chỉ để khởi tạo. Không suy từ thành tích CEC sang bảo đảm hơn WE. Population 40–80 là phương án tuning cho ngân sách này, không phải tham số nguyên bản được chứng minh tốt nhất.
- [GPSAF, Mục IV-B và V](https://arxiv.org/html/2204.04054) có bằng chứng limited-budget, nhưng chủ yếu đến 300 evaluations, số biến giữ tương đối nhỏ, và tự nêu giới hạn khi tăng kích thước dữ liệu/chi phí mô hình. Bảng III không chứng minh mọi GPSAF variant luôn hơn base; G2 là phản ví dụ. Thứ hạng giữa các biến thể là evidence trên benchmark đó, không phải dự báo trực tiếp cho UAV.
- [SCBO, Mục 4 và phụ lục](https://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf) xét cả bài ràng buộc 30/60/124 chiều; không nên hạ ưu tiên chỉ vì cho rằng BO không xử lý nổi 20 chiều. Điểm chưa chắc phù hợp ở đây là objective admission dạng bậc thang, constraints hình học vốn tính rẻ/chính xác, và overhead GP khi evaluator MILP không quá đắt. Khả năng tiết kiệm calls đáng khảo sát, khả năng tiết kiệm thời gian chưa rõ.
- [CMA-ES, Mục 6](https://arxiv.org/pdf/1604.00772) sử dụng thứ hạng và học covariance, thuận lợi cho các biến liên kết. Chất lượng học vẫn phụ thuộc tín hiệu xếp hạng; nhiều ties do mode rời rạc có thể làm suy yếu tín hiệu. Không có bằng chứng hiện tại đặt CMA-ES cao nhất trên evaluator này.
- Local Branching nguyên bản trên toàn model 5082 vị trí chưa chắc nhanh. Thiết kế restricted-neighborhood giảm tập vị trí là phần thích nghi bổ sung; nếu dùng nó để kỳ vọng cải thiện, phải gọi đúng thiết kế đó. Có thể cần đổi đồng thời hai vị trí mới tiến bộ, và có thể không cải thiện trong ngân sách hữu hạn.

### Nút thắt từ log của dự án

Đọc lại 50 map hoàn thành cho mỗi mức tải trong `GA_logs-20261013T002029Z-1-001/GA_logs/results_ue_count_sweep/runs`:

| UE | Số UE phục vụ trung bình | PRB trung bình / 135 | RF power trung bình / 10 W | Map dùng >=130 PRB |
|---|---:|---:|---:|---:|
| 50 | 49.98 | 120.46 | 7.804 | 8/50 |
| 60 | 58.10 | 133.98 | 9.514 | 47/50 |
| 70 | 62.34 | 134.92 | 9.700 | 50/50 |

Tải cao đã gần chạm quỹ tài nguyên trên các nghiệm tìm được. Đây không phải cận tối ưu hoặc bằng chứng rằng tăng service là bất khả thi. Nó cho thấy giả thuyết cải thiện phù hợp là thay vị trí/góc để giảm tài nguyên cần cho các link, đồng thời reassign UE. Bổ sung một surrogate mới tự nó không tăng quỹ tài nguyên.

Ở placement cố định, MILP đã tối ưu admission theo mô hình mode hiện có. Một bước chỉ đổi association trên cùng placement không thể bảo đảm tăng served count so với solve optimal đó. Refinement có ý nghĩa phải mở thêm placement hoặc thay đổi các quyết định chưa được tối ưu trong oracle.

Chỉ tiêu tài nguyên vẫn có hai chính sách: MILP ưu tiên PRB trước power; tầng ngoài dùng contextual resource fitness. Vì vậy kỳ vọng giảm power bằng Local Branching phải được hiểu theo mục tiêu đang giữ, không tự hứa giảm power ở cùng PRB/service và không đổi objective khi chưa được yêu cầu.

## Bài toán cần giữ khi so sánh

- Một ứng viên chứa đủ 5 UAV, mỗi UAV có `(x,y,h,theta)`: 20 biến tìm kiếm liên tục.
- Giữ nguyên `env.py`, demand UE, mode công suất/PRB, pool chung, giới hạn công suất, separation và mục tiêu MILP.
- MILP tại vị trí cố định tối ưu theo thứ tự: số UE, admitted demand, PRB, công suất.
- Tầng ngoài hiện so sánh khả thi, coverage, service, rồi resource fitness; weights có thể thay đổi theo context.
- Code và log lịch sử có thể khác cấu hình. Mỗi đợt đối chứng phải lưu cấu hình, hash và input map; không tự tái sử dụng nghiệm lịch sử nếu mô hình hoặc giao thức chấm khác.

## 1. Tám công trình sát ứng dụng UAV

| # | Công trình và nguồn | Cơ chế / ưu điểm | Hạn chế khi áp dụng ở đây | Quyết định |
|---|---|---|---|---|
| 1 | Lyu et al., **Placement Optimization of UAV-Mounted Mobile Base Stations**, IEEE Communications Letters, 2017; preprint 2016. [Bài tác giả](https://arxiv.org/abs/1612.01047) | Đặt UAV tuần tự theo spiral, ưu tiên người dùng biên; thuật toán hình học thời gian đa thức. Có ích khi tạo vị trí khởi đầu. | Mục tiêu giảm số UAV để phủ tất cả người dùng, khác bài toán cố định 5 UAV và QoS/tài nguyên rời rạc. Footprint không đủ bảo đảm service. | Dùng ý tưởng khởi tạo hoặc baseline hình học thích nghi, không chọn làm đối chứng chính. |
| 2 | Yang et al., **Joint Altitude, Beamwidth, Location and Bandwidth Optimization for UAV-Enabled Communications**, IEEE Communications Letters, 2018. [Bài tác giả](https://arxiv.org/abs/1806.02221) | Tối ưu đồng thời vị trí, độ cao, beamwidth, bandwidth và công suất uplink bằng thuật toán lặp. Sát đánh đổi hình học–anten. | Mô hình uplink và mục tiêu giảm tổng công suất khác evaluator hiện tại; không thể chép cập nhật và giữ nguyên chứng minh. | Related work bắt buộc; chưa ưu tiên tái hiện để chạy trên bộ map này. |
| 3 | Zamani et al., **Optimization of unmanned aerial vehicle augmented ultra-dense networks**, 2020. [Bài báo](https://link.springer.com/article/10.1186/s13638-020-01804-3) | Quantize vị trí rồi chọn deployment/association/power bằng MILP; có PSO. Cho phép theo dõi incumbent và bound trên mô hình hữu hạn. | Mô hình gốc có C-RAN/RRH/fronthaul và các chi phí khác. Lưới nhiều vị trí tạo MILP lớn; nghiệm tối ưu trên lưới không chứng nhận miền liên tục. | Đã có bản thích nghi. Giữ làm đối chứng, ưu tiên cải thiện cách tìm nghiệm thay vì tăng lưới ngay. |
| 4 | Zhong et al., **QoS-Compliant 3D Deployment Optimization Strategy for UAV Base Stations**, preprint 2020. [Bài tác giả](https://arxiv.org/abs/2008.03125) | GA xét phân bố demand và khả năng phục vụ khi tìm deployment. Sát hơn heuristic chỉ phủ hình học. | Gene, mục tiêu, cách suy độ cao và tài nguyên không trùng dự án. Bản `zhong_ga` hiện có là bản thích nghi. | Giữ đối chứng UAV hiện có; không dùng chênh lệch để quy riêng tác dụng WE. |
| 5 | Zhao et al., **3D placement of UAVs with optimal beamforming for multi-user communications**, Computers and Electrical Engineering, 108, 108665, 2023. [Trang công bố của trường tác giả](https://research.knu.ac.kr/en/publications/3d-placement-of-uavs-with-optimal-beamforming-for-multi-user-comm/) | ILP và hai GA thay số UAV, vị trí 3D và beamwidth. Rất liên quan đến lý do tối ưu theta. | Xét relay IoT, số flight và thời gian phục vụ/pin; khác fixed-UAV downlink PRB pool. Chưa xác minh mã chạy được trong đợt khảo sát này. | So sánh về mô hình và novelty; chưa ưu tiên port toàn bộ. |
| 6 | Rahimi et al., **3D UAV Small Cell Base Station Positioning and Resource Allocation in Cellular Network: A Stochastic Optimization Approach**, IEEE IoT Journal, 2024. [Bài tác giả](https://eprints.whiterose.ac.uk/id/eprint/205072/) | Xét bất định LoS/NLoS, association, tài nguyên và nhiễu; dùng piecewise linearization, CPLEX. | Thêm uncertainty vật lý sẽ thay đổi bài toán. Tái hiện nguyên bản không đo riêng ưu điểm cơ chế tìm kiếm trên evaluator hiện tại. | Related work về độ thực tế; hướng mở rộng sau, không phải baseline đầu tiên. |
| 7 | Premkumar & Van Scoy, **Optimal Positioning of Unmanned Aerial Vehicle (UAV) Base Stations Using Mixed-Integer Linear Programming**, Drones 9(1), 44, 2025. [Bài báo](https://www.mdpi.com/2504-446X/9/1/44) | MILP tối đa người dùng, có clustering để cân bằng chi phí và coverage. Phù hợp khảo sát khả năng mở rộng. | Không nên coi mô hình coverage của bài là tương đương phân bổ PRB/power/QoS trong dự án. Phần cluster có thể mất chi tiết demand và UE biên. | Có thể làm baseline clustering–MILP thích nghi; ưu tiên thấp hơn DE/GPSAF. |
| 8 | Geng et al., **Joint Deployment and Resource Allocation for Service Provision in Multi-UAV-Assisted Wireless Networks**, IEEE IoT Journal 11(22), 37269–37286, 2024. [Bản từ trường tác giả](https://ai.nju.edu.cn/rinc/publish/download/2024/2024JointDeployment.pdf) | BCD chia deployment và allocation; SCA giải các xấp xỉ lồi, khai thác cấu trúc bài toán. | Công thức và khởi tạo khả thi của bài gốc quan trọng. Demand admission, footprint và mode rời rạc hiện tại tạo gián đoạn; không thể giữ định lý hội tụ khi thay mô hình. | Có giá trị lý thuyết, nhưng chi phí thích nghi cao; chưa ưu tiên để có runner ổn sớm. |

## 2. Tám phương pháp tối ưu có thể làm tầng ngoài hoặc matheuristic

| # | Công trình và nguồn | Ưu điểm liên quan | Nhược điểm / việc phải thích nghi | Mức ưu tiên |
|---|---|---|---|---|
| 9 | Storn & Price, **Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces**, Journal of Global Optimization, 1997. [Bài báo](https://doi.org/10.1023/A:1008202821328) | Không cần đạo hàm; cấu trúc gọn; có thể dùng cùng evaluator và comparator để chọn parent/trial. | Cần diversity, xử lý biên/separation và ngân sách rõ ràng; có thể trì trệ trên plateau số UE. Bản có comparator riêng là DE thích nghi. | **1: triển khai DE–MILP trước.** |
| 10 | Tanabe & Fukunaga, **Improving the Search Performance of SHADE Using Linear Population Size Reduction**, CEC 2014. [Bài tác giả](https://ryojitanabe.github.io/pdf/tf-cec2014.pdf), [mã tác giả](https://ryojitanabe.github.io/publication) | L-SHADE học tham số từ lần cải thiện thành công, giảm population theo ngân sách. Là đối chứng DE có cơ sở mạnh. | Bản gốc là scalar unconstrained; success-history dùng mức cải thiện fitness. Không thể thay bằng thứ hạng tùy ý rồi gọi nguyên bản. Tham số population gốc lớn so với 1280 calls ở 20 chiều. | Sau DE pilot; dùng mã sửa lỗi 1.0.1 và công bố mọi thay đổi. |
| 11 | Hansen, **The CMA Evolution Strategy: A Tutorial**, 2016. [Tài liệu tác giả](https://arxiv.org/abs/1604.00772), [pycma](https://github.com/CMA-ES/pycma) | Học tương quan giữa 20 biến liên tục; ask/tell thuận tiện ghép MILP; không cần gradient. | Plateaus, miền khả thi rời rạc do association và symmetry UAV có thể gây khó. CMA đang có trong repo là sampler thích nghi, không phải đối chứng CMA-ES chuẩn. | **4: thêm CMA-ES–MILP độc lập nếu cần.** |
| 12 | Blank & Deb, **GPSAF: A Generalized Probabilistic Surrogate-Assisted Framework for Constrained Single- and Multi-objective Optimization**, preprint/COIN report 2022. [Bài tác giả](https://arxiv.org/abs/2204.04054) | Phối hợp tìm kiếm và surrogate, xét approximation error, probabilistic knockout tournament; có ràng buộc. Đối chứng rất sát phần sàng lọc của bạn. | Cần ánh xạ đúng objective và constraint. Không tự hỗ trợ thứ tự từ điển và weights biến thiên của dự án. Có chi phí fit/model management. | **2: GPSAF–DE–MILP là đối chứng học thuật quan trọng nhất.** |
| 13 | Regis & Shoemaker, **Combining radial basis function surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization**, Engineering Optimization 45(5), 529–555, 2013. [Triển khai và trích dẫn trong pySOT](https://github.com/dme65/pySOT/blob/master/docs/options.rst), [mã](https://github.com/dme65/pySOT) | DYCORS dùng RBF, nhiễu các tọa độ với xác suất thay đổi và cân bằng dự đoán–khoảng cách. Ít cơ chế hơn khung hiện tại. | Scalar/bound-constrained mặc định không thay thế trực tiếp comparator hiện tại. RBF có thể khó khớp các bước nhảy admission. Cần constraint handling riêng hoặc adaptation được mô tả. | Sau GPSAF; đối chứng surrogate khác họ đáng thử. |
| 14 | Eriksson & Poloczek, **Scalable Constrained Bayesian Optimization**, AISTATS 2021. [Bài báo](https://proceedings.mlr.press/v130/eriksson21a.html), [BoTorch SCBO](https://botorch.org/docs/v0.16.1/tutorials/scalable_constrained_bo) | Trust-region BO cho miền khó tìm nghiệm khả thi; phù hợp kiểm tra sample efficiency. Có tutorial có mã. | Fit GP và candidate sampling có overhead; objective của dự án dạng bậc thang và nhiều tie. Chưa có bằng chứng bù được overhead khi evaluator MILP nhanh. | Đợt hai; không chọn đầu tiên. |
| 15 | Fischetti & Lodi, **Local Branching**, Mathematical Programming 98, 23–47, 2003. [Thông tin công bố của tác giả](https://publications.polymtl.ca/25914/), [tutorial tác giả](https://site.unibo.it/operations-research/en/research/library-of-codes-and-instances-1/mic2003-lb.pdf/@@download/file/mic2003-lb.pdf) | Thêm ràng buộc khoảng cách Hamming quanh incumbent, giải sub-MILP để cải thiện. Tận dụng model và solver hiện có. | Cần incumbent khả thi và tập ứng viên hợp lý; có thể kẹt lân cận. Giới hạn lân cận không chứng nhận toàn cục. Thêm cut vào model lớn chưa chắc làm giảm chi phí build/LP. | **3: đối chứng dùng MILP làm công cụ tìm kiếm.** |
| 16 | Danna, Rothberg & Le Pape, **Exploring relaxation induced neighborhoods to improve MIP solutions**, Mathematical Programming 102, 71–90, 2005. [Bài báo](https://link.springer.com/article/10.1007/s10107-004-0518-7) | RINS tạo sub-MIP từ các biến đồng ý giữa LP relaxation và incumbent; khai thác thông tin solver. | Cần incumbent, LP solution và chiến lược chọn/fix biến. Có thể trùng một phần với primal heuristics solver. Tích hợp đúng tốn công hơn Local Branching. | Chỉ làm sau Local Branching nếu cần đối chứng MIP mạnh hơn. |

## 3. Trạng thái mã công khai và môi trường đã kiểm tra

Kiểm tra read-only trên interpreter đã dùng trong chat trước:
`C:/Users/Hoa Ly/AppData/Local/Programs/Python/Python313/python.exe`.

| Hạng mục | Kết quả |
|---|---|
| Python | 3.13.7 |
| Có sẵn | NumPy 2.5.2, SciPy 1.16.3, highspy 1.15.1, CVXPY 1.9.2, pymoo 0.6.2, torch 2.11.0 |
| Chưa tìm thấy trong interpreter này | pysamoo, pySOT, poap, botorch |
| pycma | `find_spec('cma')` trả về file `cma.py` của dự án; kết quả này không xác nhận package pycma có hoặc không được cài |

- **GPSAF:** mã [pysamoo](https://github.com/anyoptimization/pysamoo) và [ví dụ của tác giả](https://anyoptimization.com/projects/pysamoo/) đã xác minh. `setup.py` được xem đang pin `pymoo==0.6.1.1` và yêu cầu `ezmodel`, khác pymoo hiện tại. Cần môi trường tách riêng và smoke test; chưa kiểm chứng tương thích Python 3.13. Không cài đè môi trường hiện có chỉ để thử baseline.
- **L-SHADE:** tác giả có C++, MATLAB/Octave và Java, có bản sửa lỗi 1.0.1. Một bản port Python vẫn phải kiểm tra memory update, archive và population reduction. Không lấy một gói không rõ provenance rồi coi là mã gốc.
- **CMA-ES:** pycma có Python ask/tell và constraint handling. Cần tránh xung đột tên `cma` với file nội bộ, bằng cách cô lập import/runner; không cần tự ý đổi module chính.
- **DYCORS:** có mã Python trong pySOT; chưa cài và chưa kiểm chứng tương thích môi trường hiện tại.
- **SCBO:** BoTorch có tutorial chính thức; torch đã có nhưng botorch chưa có. Tutorial là nền cho implementation, không phải runner UAV sẵn dùng.
- **DE và Local Branching:** có thể xây runner riêng bằng NumPy và highspy hiện có. Đây là nhận định về phụ thuộc; chưa phải xác nhận runner đã được viết/chạy.

## 4. Cách ghép bốn lựa chọn ưu tiên

### A. DE–MILP

Giữ một population cấu hình toàn hệ thống; sinh trial bằng DE; repair theo chính sách chung; đánh giá bằng evaluator hiện có; chọn parent/trial theo comparator chung. Không dùng surrogate/WE/CMA.

Đề xuất pilot population 40 hoặc 60, chọn trên tập tuning riêng. Đây là thiết lập đề xuất phù hợp ngân sách 1280, không phải tham số chứng minh tốt nhất. Chuẩn hóa 20 biến về [0,1]. Chặn tổng số unique MILP solves gồm warmup. Giữ archive nghiệm exact khả thi và mọi trace theo calls/thời gian.

### B. GPSAF–DE–MILP

Đưa evaluator vào bài toán constrained của pysamoo; dùng DE làm base optimizer, GPSAF quyết định việc khai thác surrogate và chọn infill. Lưu forecast trước evaluation, mọi thời gian fit và số calls thật.

Khó khăn quan trọng là thứ tự coverage/service/resource và context weights biến thiên. Để dùng scalar API, cần một mã hóa có cận chứng minh giữ đúng ưu tiên trên nghiệm exact, rồi xác minh bằng unit tests; không chọn penalty tùy ý. Hoặc sửa comparator và ghi rõ GPSAF thích nghi. Đổi weights làm objective đổi theo thời gian: lưu primitives và rescore nhãn theo context chung trước fit, hoặc chạy một cohort weights cố định cho tất cả phương pháp. Không tự thay mục tiêu MILP.

### C. Local Branching–candidate MILP

Với biến chọn vị trí y_s và incumbent có tập vị trí I, thêm:

`sum_{s in I}(1-y_s) + sum_{s not in I}y_s <= 2q`.

Vì chọn đúng 5 vị trí, q=1 cho phép đổi tối đa một vị trí đang chọn sang một vị trí khác trong tập ứng viên. Khởi đầu q=1, mở q=2 khi không cải thiện, đa khởi tạo để giảm phụ thuộc nghiệm đầu.

Phải phân biệt hai cách triển khai: (a) cut Local Branching trên cùng tập ứng viên; (b) restricted-neighborhood MILP với ít vị trí hơn. Cách (b) có thể giảm kích thước mạnh hơn nhưng là thiết kế thích nghi bổ sung, không tự động là Local Branching nguyên bản.

Nếu dùng tập ứng viên gồm vị trí liên tục từ seed, báo cáo riêng với lưới thuần. Có thể tạo seed bằng cùng chính sách hình học/đa khởi tạo cho mọi thuật toán. Không lấy nghiệm WE làm warmstart chỉ cho baseline rồi coi đó là chạy độc lập; nếu đánh giá khả năng refinement từ WE, phải gọi đúng là hybrid và tính toàn bộ thời gian seed.

MILP trong lân cận tối ưu theo mục tiêu MILP chung, không nhất thiết theo resource score contextual bên ngoài. Giữ sự khác biệt này trong diễn giải và báo cáo tài nguyên theo từng trục.

### D. CMA-ES–MILP

Dùng pycma độc lập, với covariance/step-size/restart theo thuật toán chuẩn đã chọn. Mọi ứng viên dùng cùng MILP; constraint handling và archive cuối phải được mô tả. So sánh với toàn hệ thống của bạn đo giá trị toàn khung; không cô lập riêng tác dụng WE.

## 5. Pilot để xác định phương pháp nào cho kết quả ổn

Không chọn baseline để bảo đảm WE thắng. Chọn phương pháp có nền tảng, mô hình phù hợp và triển khai kiểm chứng được.

1. Smoke test trên map nhỏ: kiểm tra budget, feasible archive, solver status, objective order, cache và khôi phục sau ngắt. Không dùng smoke test để kết luận chất lượng.
2. Pilot trên 5 map 50 UE và 5 map 70 UE, mỗi map 3 seed thuật toán, định danh cố định trước khi xem kết quả. Cùng input và config cho mọi phương pháp.
3. Ghi các mốc 320, 640, 1280 unique MILP calls. Với Local Branching, một sub-MIP không tương đương một fixed-placement MILP: so sánh chính theo wall time cùng máy, kèm model size, solver time và sub-MIP count.
4. Đo số UE của incumbent exact khả thi, admitted demand, thời gian đến nghiệm khả thi đầu tiên, xác suất đạt ngưỡng service đặt trước, PRB/power khi cùng service, độ phân tán qua seed, tổng thời gian và overhead.
5. Không loại run chưa đạt ngưỡng hoặc run không có feasible incumbent khỏi thống kê; ghi failed-to-target/censored và failure rõ ràng.
6. Dùng pilot để kiểm tra độ ổn định và chọn tham số; sau đó đóng cấu hình và chạy bộ map khác cho kết luận cuối. Không công bố thắng trên pilot tuning như kết quả test độc lập.

Tổng mỗi phương pháp trong pilot là 10 map × 3 seed = 30 run. Đây là thiết kế đề xuất, chưa được thực hiện. Ba seed chỉ giúp sàng lọc ban đầu; cần mở rộng nếu kết quả dao động lớn.

## 6. Kết luận sử dụng

- **Muốn có runner gọn và ít phụ thuộc nhất:** DE–MILP.
- **Muốn đối chứng sát đóng góp surrogate và xác suất:** GPSAF–DE–MILP.
- **Muốn một phương pháp thuộc họ MILP có cơ hội ra incumbent tốt hơn Grid hiện tại:** Local Branching / restricted-neighborhood MILP, báo cáo đúng phần thích nghi.
- **Muốn đối chứng chuẩn cho tìm kiếm 20 chiều:** CMA-ES–MILP.
- **Muốn thêm surrogate khác họ:** DYCORS sau khi giải quyết objective/constraint mapping.

Chưa cài package, viết runner mới hay chạy benchmark trong đợt khảo sát này. Mã thuật toán chính và mục tiêu MILP không bị sửa.
