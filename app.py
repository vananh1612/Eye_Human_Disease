import streamlit as st  # type: ignore
import tensorflow as tf
import tempfile
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import os
import io

@st.cache_resource
def load_model():
    # Load the pre-trained model
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    return model


def model_prediction(test_image_path):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(
        test_image_path, target_size=(224, 224)
    )

    x = tf.keras.utils.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)

    return np.argmax(prediction)


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Identification"]
)


# Trang Home
if app_mode == "Home":
    # image_path = "home_page.jpeg"
    # st.image(image_path,use_column_width=True)
    st.markdown(
        """
## **Nền tảng Phân Tích Ảnh OCT Võng Mạc**

#### **Chào mừng bạn đến với Nền tảng Phân Tích OCT Võng Mạc**

**Chụp cắt lớp quang học (OCT)** là một kỹ thuật hình ảnh hiện đại, cung cấp hình ảnh cắt ngang có độ phân giải cao của võng mạc, cho phép phát hiện sớm và theo dõi các bệnh lý võng mạc khác nhau. Mỗi năm, có hơn 30 triệu lượt chụp OCT được thực hiện, hỗ trợ chẩn đoán và điều trị các bệnh lý về mắt có thể gây mất thị lực như tân mạch hắc mạc (CNV), phù hoàng điểm do đái tháo đường (DME) và thoái hóa điểm vàng do tuổi tác (AMD).

##### **Tại sao OCT quan trọng?**
OCT là một công cụ không xâm lấn thiết yếu trong nhãn khoa, giúp phát hiện những bất thường ở võng mạc. Trên nền tảng này, chúng tôi hướng đến việc đơn giản hóa quá trình phân tích và diễn giải ảnh OCT, giảm tải công việc cho bác sĩ và nâng cao độ chính xác trong chẩn đoán thông qua các mô hình phân tích tự động tiên tiến.

---

#### **Tính năng nổi bật của nền tảng**

- **Phân tích hình ảnh tự động**: Ứng dụng các mô hình học máy hiện đại để phân loại ảnh OCT thành các nhóm: **Bình thường**, **CNV**, **DME** và **Drusen**.
- **Hình ảnh cắt lớp võng mạc chất lượng cao**: Hỗ trợ bác sĩ đưa ra quyết định lâm sàng chính xác thông qua ảnh minh họa rõ ràng của võng mạc bình thường và bệnh lý.
- **Quy trình làm việc tối ưu**: Tải lên, phân tích và xem kết quả ảnh OCT chỉ trong vài bước đơn giản.

---

#### **Hiểu về các bệnh lý võng mạc thông qua ảnh OCT**

1. **Tân mạch hắc mạc (CNV)**
   - Màng tân mạch kèm theo dịch dưới võng mạc.

2. **Phù hoàng điểm do đái tháo đường (DME)**
   - Võng mạc dày lên, xuất hiện dịch trong võng mạc.

3. **DRUSEN (Giai đoạn sớm của AMD)**
   - Sự hiện diện của nhiều cặn Drusen dưới biểu mô sắc tố võng mạc.

4. **Võng mạc bình thường(NORMAL)**
   - Cấu trúc hoàng điểm được bảo tồn, không có dịch hoặc phù.

---

#### **Về tập dữ liệu**

Tập dữ liệu của chúng tôi gồm **84.495 ảnh OCT độ phân giải cao** (định dạng JPEG), được chia thành các tập **huấn luyện, kiểm thử và xác thực**, tương ứng với 4 nhóm chính:
- **NORMAL**
- **CNV**
- **DME**
- **DRUSEN**

Mỗi ảnh đều được kiểm định nhiều lớp bởi các chuyên gia nhằm đảm bảo độ chính xác trong phân loại bệnh. Các ảnh được thu thập từ nhiều trung tâm y tế uy tín trên toàn thế giới, phản ánh sự đa dạng trong đối tượng bệnh nhân và các dạng bệnh lý võng mạc.

---

#### **Bắt đầu ngay**

- **Tải ảnh OCT lên**: Bắt đầu bằng cách tải các ảnh OCT của bạn lên để phân tích.
- **Khám phá kết quả**: Xem phân loại ảnh và thông tin chẩn đoán chi tiết.
- **Tìm hiểu thêm**: Hiểu sâu hơn về các bệnh võng mạc và vai trò của OCT trong chẩn đoán.

---

#### **Liên hệ với chúng tôi**

Bạn có câu hỏi hoặc cần hỗ trợ? [Liên hệ với đội ngũ hỗ trợ của chúng tôi](#) để biết thêm thông tin về cách sử dụng nền tảng hoặc tích hợp vào quy trình lâm sàng của bạn.
"""
    )

# Trang About
elif app_mode == "About":
    st.header("Giới thiệu")
    st.markdown(
        """
          #### Về Tập Dữ Liệu
Chụp cắt lớp quang học võng mạc (OCT) là một kỹ thuật hình ảnh dùng để thu nhận các lát cắt cắt ngang có độ phân giải cao của võng mạc ở bệnh nhân còn sống.  
Mỗi năm có khoảng 30 triệu lượt chụp OCT được thực hiện, và việc phân tích và diễn giải những ảnh này tiêu tốn một lượng thời gian đáng kể.

- (A) (Góc ngoài cùng bên trái) Tân mạch hắc mạc (CNV) với màng tân mạch (các mũi tên trắng) và dịch dưới võng mạc liên quan (các mũi tên).  
- (Bên trái giữa) Phù hoàng điểm do đái tháo đường (DME) với hiện tượng dày võng mạc và có dịch trong võng mạc (các mũi tên).  
- (Bên phải giữa) Nhiều cặn drusen (mũi tên) xuất hiện trong giai đoạn sớm của thoái hóa điểm vàng do tuổi tác (AMD).  
- (Ngoài cùng bên phải) Võng mạc bình thường với cấu trúc hoàng điểm được bảo tồn và không có dịch/phù võng mạc.

---

#### Nội dung
Tập dữ liệu được tổ chức thành 3 thư mục (train, test, val) và chứa các thư mục con tương ứng với từng loại ảnh (NORMAL, CNV, DME, DRUSEN).  
- Tổng cộng có 84.495 ảnh X-Quang OCT (định dạng JPEG) thuộc 4 nhóm bệnh: NORMAL, CNV, DME, DRUSEN.

- Các ảnh được gán nhãn theo định dạng: (bệnh)-(ID bệnh nhân ngẫu nhiên)-(số thứ tự ảnh của bệnh nhân đó), và được chia thành 4 thư mục: CNV, DME, DRUSEN, và NORMAL.

Các ảnh OCT (chụp bằng máy Spectralis OCT, Heidelberg Engineering, Đức) được tuyển chọn từ các bộ dữ liệu hồi cứu của bệnh nhân trưởng thành từ các cơ sở y tế sau:  
- Viện Mắt Shiley – Đại học California San Diego  
- Quỹ Nghiên cứu Võng mạc California  
- Trung tâm Y tế Ophthalmology Associates  
- Bệnh viện Nhân dân số 1 Thượng Hải  
- Trung tâm Mắt Đồng Nhân Bắc Kinh  

Thời gian thu thập từ ngày 1/7/2013 đến 1/3/2017.

Trước khi đưa vào huấn luyện, mỗi ảnh được xử lý qua một hệ thống phân loại nhiều tầng gồm các chuyên gia có mức độ kinh nghiệm tăng dần nhằm xác minh và chỉnh sửa nhãn ảnh.  
Ban đầu, ảnh được gán nhãn theo chẩn đoán gần nhất của bệnh nhân.  
- **Tầng thứ nhất** gồm các sinh viên đại học và y khoa đã vượt qua khóa đào tạo đọc ảnh OCT. Họ tiến hành kiểm tra chất lượng ban đầu và loại bỏ các ảnh có nhiễu nghiêm trọng hoặc độ phân giải thấp.  
- **Tầng thứ hai** gồm 4 bác sĩ nhãn khoa độc lập đánh giá lại các ảnh đã qua tầng một. Họ ghi nhận sự hiện diện hay không của CNV (hoạt động hoặc xơ hóa dưới võng mạc), phù hoàng điểm, drusen và các bệnh lý khác hiển thị trên ảnh OCT.  
- **Tầng thứ ba** gồm hai chuyên gia võng mạc cấp cao, mỗi người có hơn 20 năm kinh nghiệm lâm sàng, xác minh nhãn chính xác cuối cùng cho từng ảnh.

Quá trình lựa chọn và phân tầng dữ liệu được trình bày chi tiết theo sơ đồ CONSORT tại Hình 2B.  
Để giảm thiểu sai sót từ con người trong quá trình gán nhãn, một tập hợp con gồm 993 ảnh được đánh giá độc lập bởi hai bác sĩ nhãn khoa; mọi trường hợp bất đồng sẽ được phân xử bởi một chuyên gia võng mạc cao cấp.
"""
    )

# Trang Disease Identification 
elif app_mode == "Disease Identification":
    st.header("Chào mừng đến với trang Phân loại bệnh lý võng mạc")

    # Chọn phương thức nhập ảnh
    st.subheader("🔍 Chọn cách nhập ảnh OCT")
    input_method = st.radio(
        "Bạn muốn sử dụng phương pháp nào?",
        ["📁 Tải ảnh từ máy", "📷 Chụp ảnh từ webcam"],
    )

    test_image = None
    captured_image = None
    temp_file_path = None

    if input_method == "📁 Tải ảnh từ máy":
        test_image = st.file_uploader("Tải ảnh từ thiết bị", type=["jpg", "jpeg", "png"])
    elif input_method == "📷 Chụp ảnh từ webcam":
        captured_image = st.camera_input("Chụp ảnh trực tiếp từ webcam")

    # Chọn ảnh phù hợp (ưu tiên ảnh upload nếu có)
    input_image = test_image if test_image is not None else captured_image

    if input_image is not None:
        from PIL import Image
        
        # Đọc ảnh từ input
        image = Image.open(input_image)

        # Nếu ảnh đến từ webcam, lật lại để đúng chiều
        if input_method == "📷 Chụp ảnh từ webcam":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Lưu ảnh vào file tạm để đưa vào model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file, format="JPEG")
            temp_file_path = tmp_file.name

        # Hiển thị ảnh (đã xử lý nếu có) cho người dùng xem
        st.image(temp_file_path, caption="Ảnh được sử dụng để phân tích", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                result_index = model_prediction(temp_file_path)
                class_name = ["CNV", "DME", "DRUSEN", "NORMAL"]

            st.success("Model dự đoán ảnh này thuộc loại: **{}**".format(class_name[result_index]))

            with st.expander("Đọc thêm"):
                st.write("Kết quả phân tích ảnh OCT:")

                if result_index == 0:
                    st.write("Ảnh chụp OCT cho thấy *CNV (tân mạch hắc mạc) với dịch dưới võng mạc.*")
                    st.image(input_image)
                    st.markdown(cnv)

                elif result_index == 1:
                    st.write("Ảnh chụp OCT cho thấy *DME với tình trạng dày võng mạc và dịch trong võng mạc.*")
                    st.image(input_image)
                    st.markdown(dme)

                elif result_index == 2:
                    st.write("Ảnh chụp OCT cho thấy *các lắng đọng DRUSEN trong giai đoạn đầu của thoái hóa điểm vàng (AMD).*")
                    st.image(input_image)
                    st.markdown(drusen)

                elif result_index == 3:
                    st.write("Ảnh chụp OCT cho thấy *võng mạc bình thường với hình dạng hố hoàng điểm được bảo toàn.*")
                    st.image(input_image)
                    st.markdown(normal)


