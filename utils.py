import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_images(df, image_dir="dataset/images", num_retries=3, timeout=10):
    """
    Downloads product images from the URLs in the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'sample_id' and 'image_link' columns.
        image_dir (str): Directory where images will be saved.
        num_retries (int): Number of retries for failed downloads.
        timeout (int): Timeout for each request in seconds.
    
    Returns:
        List[str]: List of local image paths corresponding to each sample.
    """
    os.makedirs(image_dir, exist_ok=True)
    local_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        sample_id = row["sample_id"]
        url = row["image_link"]
        img_path = os.path.join(image_dir, f"{sample_id}.jpg")

        # Skip if already downloaded
        if os.path.exists(img_path):
            local_paths.append(img_path)
            continue

        success = False
        for attempt in range(num_retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img.save(img_path, "JPEG")
                    success = True
                    break
            except Exception as e:
                print(f"Retry {attempt+1}/{num_retries} for {url} failed: {e}")
        
        if not success:
            print(f"⚠ Failed to download {url}, using placeholder.")
            # Create a blank placeholder image if download fails
            img = Image.new("RGB", (224, 224), (255, 255, 255))
            img.save(img_path, "JPEG")

        local_paths.append(img_path)

    return local_paths


def preprocess_image(image_path, image_size=(224, 224)):
    """
    Loads and resizes an image for embedding generation.
    Returns a PIL image or tensor ready for model input.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        # fallback to blank white image
        img = Image.new("RGB", image_size, (255, 255, 255))
    
    img = img.resize(image_size)
    return img