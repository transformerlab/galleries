import os
import json
import yaml
import requests
import time

url_check_cache = {}

def check_url_reachable(url, timeout=10, max_retries=3):
    if not url or url == "?":
        return True
    
    if url in url_check_cache:
        return url_check_cache[url]
    
    # Use a standard browser User-Agent to reduce 429s
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            # Small base delay to prevent flooding
            time.sleep(1.0)
            
            response = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
            
            # Handle Rate Limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    print(f"      ⏳ Rate limit (429). Pausing {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"      ⚠️  Rate limit persists. Assuming valid: {url}")
                    url_check_cache[url] = True
                    return True
            
            if response.status_code == 404:
                url_check_cache[url] = False
                return False
            
            # Fallback to GET if HEAD fails (some servers block HEAD)
            if response.status_code >= 400:
                time.sleep(1.0)
                response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers)
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 15 * (attempt + 1)
                        print(f"      ⏳ Rate limit (429) on GET. Pausing {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"      ⚠️  Rate limit persists. Assuming valid: {url}")
                        url_check_cache[url] = True
                        return True
                
                if response.status_code == 404:
                    url_check_cache[url] = False
                    return False
                
                # Check for "Soft 404" in HuggingFace body
                if 'huggingface.co' in url and response.status_code == 200:
                    content = response.text.lower()
                    if '404' in content or 'not found' in content or 'repository not found' in content:
                        url_check_cache[url] = False
                        return False
            
            result = response.status_code < 400
            url_check_cache[url] = result
            return result
            
        except requests.RequestException:
            # On network error, retry. If out of retries, fail.
            if attempt == max_retries - 1:
                print(f"      ⚠️  Network error max retries. Marking invalid: {url}")
                url_check_cache[url] = False
                return False
            time.sleep(2)
    
    url_check_cache[url] = False
    return False

def validate_model_urls(model):
    resources = model.get('resources', {})
    canonical_url = resources.get('canonicalUrl')
    download_url = resources.get('downloadUrl')
    
    model_name = model.get('name', 'Unknown')
    model_id = model.get('id', model.get('uniqueID', 'Unknown'))
    
    print(f"   👉 Checking: {model_name}")
    
    valid = True
    
    if canonical_url and canonical_url != "?":
        if not check_url_reachable(canonical_url):
            print(f"      ❌ Unreachable canonicalUrl: {canonical_url}")
            valid = False
    
    if download_url and download_url != "?":
        if not check_url_reachable(download_url):
            print(f"      ❌ Unreachable downloadUrl: {download_url}")
            valid = False
            
    return valid

def save_gallery_files(valid_models, parent_dir):
    gallery_path = os.path.join(parent_dir, 'model-gallery.json')
    group_gallery_path = os.path.join(parent_dir, 'model-group-gallery.json')
    
    if not os.path.exists(gallery_path):
        print(f"   📄 Generating '{gallery_path}'...")
        try:
            with open(gallery_path, 'w') as f:
                json.dump(valid_models, f, indent=4)
        except Exception as e:
            print(f"   ❌ Failed to create '{gallery_path}': {e}")

    if not os.path.exists(group_gallery_path):
        print(f"   📄 Generating '{group_gallery_path}'...")
        model_groups_dir = os.path.join(parent_dir, 'model-groups')
        model_groups = {}

        if os.path.exists(model_groups_dir):
            for file in os.listdir(model_groups_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(model_groups_dir, file), 'r') as f:
                            group_data = json.load(f)
                            group_data["models"] = []
                            model_groups[group_data["name"]] = group_data
                    except Exception:
                        pass
        
        for model in valid_models:
            group_name = model.get("model_group")
            if group_name and group_name in model_groups:
                model_groups[group_name]["models"].append(model)

        try:
            with open(group_gallery_path, 'w') as f:
                json.dump(list(model_groups.values()), f, indent=4)
        except Exception as e:
            print(f"   ❌ Failed to create '{group_gallery_path}': {e}")

def check_directory(directory='models'):
    print(f'Validating URLs in {directory} directory...')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    directory_path = os.path.join(parent_dir, directory)

    if not os.path.exists(directory_path):
        return

    files = [f for f in os.listdir(directory_path) if (f.endswith('.json') or f.endswith('.yaml'))]
    files.sort()
    
    total_models = 0
    failed_models = 0
    valid_models = []

    for file in files:
        # print(f" 📁 Scanning {file}...") 
        file_path = os.path.join(directory_path, file)
        
        with open(file_path, 'r') as f:
            if file.endswith('.yaml'):
                content = yaml.load(f, Loader=yaml.FullLoader)
            else:
                content = json.load(f)
            
            items = content if isinstance(content, list) else [content]
            
            for item in items:
                total_models += 1
                if validate_model_urls(item):
                    valid_models.append(item)
                else:
                    failed_models += 1
                    
    print("-" * 30)
    print(f"Total models checked: {total_models}")
    
    save_gallery_files(valid_models, parent_dir)

    if failed_models > 0:
        print(f"❌ Found {failed_models} models with unreachable URLs.")
        exit(1)
    else:
        print(f"✅ All URLs are reachable.")

if __name__ == "__main__":
    check_directory('models')