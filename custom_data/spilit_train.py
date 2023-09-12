import os
import random
import shutil

# Đường dẫn tới thư mục chứa tập dữ liệu train
train_folder = 'custom_data\\train\\images'

test_folder = 'custom_data\\test\\images'

# Đường dẫn tới thư mục chứa tập dữ liệu nhãn train
train_label_folder = 'custom_data\\data\\train\\labels'

# Đường dẫn tới thư mục chứa tập dữ liệu validation (nếu chưa có, bạn có thể tạo mới)
val_folder = 'custom_data\\data\\val\\images'

# Đường dẫn tới thư mục chứa tập dữ liệu nhãn validation (nếu chưa có, bạn có thể tạo mới)
val_label_folder = 'custom_data\\data\\val\\labels'

# Tỉ lệ dữ liệu chuyển từ train sang validation (ở đây là 30%)
validation_ratio = 0.3

# Lấy danh sách các file trong thư mục train
files = os.listdir(train_folder)

# Số lượng mẫu dữ liệu sẽ chuyển sang validation
num_validation_samples = int(len(files) * validation_ratio)

# Sử dụng random.sample để lấy ngẫu nhiên các file để chuyển sang validation
validation_files = random.sample(files, num_validation_samples)
print('total: ',len(files),' xample')
# Di chuyển các file dữ liệu và nhãn sang thư mục validation
# for file in validation_files:
#     src_data = os.path.join(train_folder, file)
#     src_label = os.path.join(train_label_folder, file.replace(".jpg", ".txt"))  # Giả sử nhãn có định dạng .txt
#     print(src_data)
    
#     dst_data = os.path.join(val_folder, file)
#     dst_label = os.path.join(val_label_folder, file.replace(".jpg", ".txt"))
    
#     print(dst_data)
#     shutil.move(src_data, dst_data)
#     shutil.move(src_label, dst_label)

# print(f"{len(validation_files)} mẫu dữ liệu và nhãn đã được chuyển sang tập validation.")
