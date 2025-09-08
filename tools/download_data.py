import requests
import mimetypes
import os
from tqdm import tqdm
import time

def download_with_progress(url, save_path, headers=None, cookies=None):
    """
    Tải file với thanh tiến trình
    """
    try:
        # Gửi HEAD request để lấy thông tin file size
        head_response = requests.head(url, headers=headers, cookies=cookies, allow_redirects=True)
        total_size = int(head_response.headers.get('content-length', 0))
        
        # Gửi GET request với stream=True để tải từng chunk
        response = requests.get(url, headers=headers, cookies=cookies, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            # Tự động xác định phần mở rộng file
            content_type = response.headers.get('Content-Type')
            print(f"Content-Type: {content_type}")
            
            file_extension = mimetypes.guess_extension(content_type)
            if file_extension:
                destination_with_extension = f"{save_path}{file_extension}"
            else:
                destination_with_extension = save_path
            
            # Hiển thị thông tin file
            print(f"Tải file: {destination_with_extension}")
            if total_size > 0:
                print(f"Kích thước: {total_size / (1024*1024):.2f} MB")
            
            # Tạo thanh tiến trình
            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc="Đang tải",
                ncols=100
            )
            
            downloaded_size = 0
            start_time = time.time()
            
            # Tải file theo chunk và cập nhật tiến trình
            with open(destination_with_extension, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        progress_bar.update(len(chunk))
                        
                        # Tính tốc độ tải (mỗi 50 chunks)
                        if downloaded_size % (8192 * 50) == 0:
                            elapsed_time = time.time() - start_time
                            if elapsed_time > 0:
                                speed = downloaded_size / elapsed_time / 1024  # KB/s
                                progress_bar.set_postfix({
                                    'Tốc độ': f'{speed:.1f} KB/s',
                                    'ETA': f'{(total_size - downloaded_size) / (speed * 1024):.1f}s' if speed > 0 else 'N/A'
                                })
            
            progress_bar.close()
            
            elapsed_time = time.time() - start_time
            avg_speed = downloaded_size / elapsed_time / 1024 if elapsed_time > 0 else 0
            
            print(f"\n✅ Tải file thành công!")
            print(f"📁 Đã lưu tại: {destination_with_extension}")
            print(f"📊 Tổng dung lượng: {downloaded_size / (1024*1024):.2f} MB")
            print(f"⏱️  Thời gian: {elapsed_time:.2f} giây")
            print(f"🚀 Tốc độ trung bình: {avg_speed:.1f} KB/s")
            
            return destination_with_extension
            
        else:
            print(f"❌ Lỗi tải file. Status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối: {e}")
        return None
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return None

def main():
    # Cấu hình
    file_url = 'https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBDAb7v0UgI8iIAExQUJq62Q?e=fIY3zh'
    save_path = 'data/coco.zip'  # Không cần phần mở rộng, sẽ tự động thêm
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        # 'Cookie': 'rtFa=...', # Thêm cookies nếu cần
    }
    
    print("🔄 Đang kết nối đến SharePoint...")
    
    # Bước 1: Gửi yêu cầu ban đầu để lấy redirect URL và cookies
    try:
        res = requests.get(file_url, headers=headers, allow_redirects=True)
        print(f"📡 Status code: {res.status_code}")
        
        if res.status_code == 200:
            # Lấy redirect url & cookies
            new_url = res.url
            cookies = res.cookies.get_dict()
            
            # Cập nhật cookies từ tất cả redirects
            for r in res.history:
                cookies.update(r.cookies.get_dict())
            
            print(f"🔗 Redirect URL: {new_url}")
            print(f"🍪 Cookies: {len(cookies)} cookies được tìm thấy")
            
            # Chuyển đổi URL để tải trực tiếp
            # Bạn có thể cần điều chỉnh logic này tùy theo cấu trúc URL của SharePoint
            download_url = 'https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/download.aspx?UniqueId=3612005b%2D0800%2D4ff2%2Dbdbc%2D46186081ee42'
            
            # Bước 2: Tải file với thanh tiến trình
            print("\n🚀 Bắt đầu tải file...")
            result = download_with_progress(download_url, save_path, headers, cookies)
            
            if result:
                print(f"\n🎉 Hoàn thành! File đã được lưu tại: {result}")
            else:
                print("\n❌ Tải file thất bại!")
                
        else:
            print(f"❌ Không thể kết nối đến SharePoint. Status code: {res.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối: {e}")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")

if __name__ == "__main__":
    main()