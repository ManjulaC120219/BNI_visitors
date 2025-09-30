import pandas as pd
import numpy as np
import os, re
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from datetime import datetime, date
import warnings
from paddleocr import PaddleOCR
import cv2
from collections import defaultdict
import tempfile
from difflib import SequenceMatcher

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set environment variables before importing PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_mkldnn"] = "0"

# Global OCR instance - will be initialized lazily
ocr_instance = None

# ========= CONFIG =========
COLUMNS = ['Name', 'Company Name', 'Category', 'Invited by', 'Fees', 'Payment Mode', 'Date','Note']


def simple_preprocess(image_path):
    """Enhanced preprocessing for better OCR"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Light bilateral filtering
    filtered = cv2.bilateralFilter(image, 5, 50, 50)

    # Slight sharpening for handwritten text
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)

    return sharpened


def save_temp_image(image_array):
    """Save image to temporary file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(temp_file.name, image_array)
    return temp_file.name


def extract_data_from_image_v2(image_path):
    """
    Extract 8 columns: Name, Company Name, Category, Invited by, Fees, Payment Mode, Date, Note
    """
    print("DEBUG: Starting 8-column extraction")

    try:
        # Enhanced preprocessing
        processed = simple_preprocess(image_path)
        temp_path = save_temp_image(processed)

        # Multiple OCR approaches
        all_ocr_items = []

        # Approach 1: Ultra-aggressive for handwritten
        try:
            ocr1 = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.01,
                det_db_box_thresh=0.03,
                det_db_unclip_ratio=3.5,
                rec_batch_num=20
            )
            result1 = ocr1.ocr(temp_path, cls=True)
            if result1 and result1[0]:
                all_ocr_items.extend(result1[0])
                print(f"DEBUG: Ultra-aggressive OCR found {len(result1[0])} items")
        except Exception as e:
            print(f"DEBUG: Ultra-aggressive OCR failed: {e}")

        # Approach 2: Standard OCR
        try:
            ocr2 = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.25,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.5
            )
            result2 = ocr2.ocr(image_path, cls=True)
            if result2 and result2[0]:
                # Add unique items based on text similarity
                existing_texts = {item[1][0].strip().lower() for item in all_ocr_items if len(item) >= 2}
                for item in result2[0]:
                    if len(item) >= 2:
                        text = item[1][0].strip().lower()
                        if text not in existing_texts:
                            all_ocr_items.append(item)
                            existing_texts.add(text)

                print(f"DEBUG: Total after standard merge: {len(all_ocr_items)}")
        except Exception as e:
            print(f"DEBUG: Standard OCR failed: {e}")

        # Approach 3: Extremely aggressive bottom area processing for handwritten
        try:
            img = cv2.imread(image_path)
            if img is not None:
                height = img.shape[0]
                # Process bottom 30% where handwritten entries are located
                bottom_crop = img[int(height * 0.7):, :]

                if bottom_crop.size > 0:
                    # Apply additional preprocessing for handwritten text
                    bottom_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
                    bottom_enhanced = cv2.bilateralFilter(bottom_gray, 9, 75, 75)

                    # Save enhanced bottom crop
                    temp_bottom = save_temp_image(bottom_enhanced)

                    # Extremely aggressive OCR for handwritten
                    ocr3 = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False,
                        show_log=False,
                        det_db_thresh=0.001,
                        det_db_box_thresh=0.005,
                        det_db_unclip_ratio=5.0,
                        rec_batch_num=30
                    )
                    result3 = ocr3.ocr(temp_bottom, cls=True)

                    if result3 and result3[0]:
                        # Adjust coordinates back to full image
                        for item in result3[0]:
                            if len(item) >= 2:
                                poly = item[0]
                                for point in poly:
                                    point[1] += int(height * 0.7)

                        all_ocr_items.extend(result3[0])
                        print(f"DEBUG: Ultra-aggressive bottom processing found {len(result3[0])} items")

                    os.unlink(temp_bottom)

                # ADDITIONAL: Try the very bottom 15% with different preprocessing
                bottom_5_crop = img[int(height * 0.85):, :]

                if bottom_5_crop.size > 0:
                    # Enhance contrast and sharpen
                    bottom_5_gray = cv2.cvtColor(bottom_5_crop, cv2.COLOR_BGR2GRAY)
                    # Adaptive thresholding to pull out faint handwriting
                    thresh = cv2.adaptiveThreshold(bottom_5_gray, 255,
                                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 15, 10)

                    kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])
                    sharpened = cv2.filter2D(thresh, -1, kernel)

                    temp_bottom_5 = save_temp_image(sharpened)

                    ocr5 = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False,
                        show_log=False,
                        det_db_thresh=0.0002,
                        det_db_box_thresh=0.0005,
                        det_db_unclip_ratio=6.0,
                    )
                    result5 = ocr5.ocr(temp_bottom_5, cls=True)

                    if result5 and result5[0]:
                        for item in result5[0]:
                            if len(item) >= 2:
                                for point in item[0]:
                                    point[1] += int(height * 0.85)
                        all_ocr_items.extend(result5[0])
                        print(f"DEBUG: Bottom 15% processing found {len(result5[0])} additional items")

                    os.unlink(temp_bottom_5)

        except Exception as e:
            print(f"DEBUG: Bottom processing failed: {e}")

        os.unlink(temp_path)

    except Exception as e:
        print(f"DEBUG: Fallback OCR: {e}")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_path, cls=True)
        all_ocr_items = result[0] if result and result[0] else []

    if not all_ocr_items:
        raise RuntimeError("No OCR results")

    # Parse OCR items with very lenient filtering
    ocr_items = []
    for line in all_ocr_items:
        if line and len(line) >= 2:
            poly, (text, score) = line[0], line[1]

            # Very lenient confidence threshold
            if score < 0.05:
                continue

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            # Minimal size filtering
            if width < 2 or height < 2:
                continue

            ocr_items.append({
                'text': text.strip(),
                'score': float(score),
                'x': sum(xs) / len(xs),
                'y': sum(ys) / len(ys),
                'width': width,
                'height': height
            })

    #print(f"DEBUG: Parsed {len(ocr_items)} OCR items")

    # Sort by Y coordinate
    ocr_items.sort(key=lambda x: x['y'])

    def is_person_name(text):
        """Improved name detection"""
        text = text.strip()

        if len(text) < 2:
            return False

        lower = text.lower()

        # Reject known column headers
        headers = ['name', 'company', 'category', 'invited', 'fees', 'payment', 'mode', 'date', 'note']
        if any(h in lower for h in headers):
            return False

        # Reject numeric or time-like values
        if re.match(r'^\d+$', lower) or re.match(r'^\d{1,2}[:.]\d{1,2}', lower):
            return False

        # Reject date patterns
        if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text):
            return False

        # Must contain at least 2 letters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < 2:
            return False

        return True

    def is_company_name(text):
        """Detect company names"""
        text = text.strip()

        if len(text) < 2:
            return False

        # Company indicators
        company_indicators = ['ltd', 'inc', 'corp', 'pvt', 'llc', 'llp', 'co', 'industries', 'technologies']
        lower = text.lower()

        # Check for company indicators
        if any(indicator in lower for indicator in company_indicators):
            return True

        # Check for all caps (common in company names)
        if text.isupper() and len(text) >= 3:
            return True

        return False

    def is_fees_amount(text):
        """Enhanced fees detection"""
        text = text.strip()

        # Clean the text
        clean = text.replace('₹', '').replace('Rs', '').replace(',', '').replace(' ', '')

        # Direct numeric check
        if clean.replace('.', '').isdigit():
            try:
                amount = float(clean)
                return 0 <= amount <= 999999
            except:
                return False

        # Handle OCR errors in amounts
        numeric_match = re.search(r'\d{2,}', clean)
        if numeric_match:
            try:
                amount = float(numeric_match.group())
                return 0 <= amount <= 999999
            except:
                return False

        return False

    def is_date_format(text):
        """Detect date values"""
        patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}',
            r'[A-Za-z]{3,9}\s+\d{1,2}[,\s]+\d{2,4}'
        ]
        return any(re.search(pattern, text.strip()) for pattern in patterns)

    def is_payment_mode(text):
        """Detect payment mode"""
        modes = ['cash', 'online', 'upi', 'card', 'cheque', 'bank', 'transfer', 'done', 'cosh', 'cagh']
        return text.lower().strip() in modes

    # Find table start
    table_start_y = 0
    for item in ocr_items:
        if 'name' in item['text'].lower() and len(item['text']) < 15:
            table_start_y = item['y'] + 15
            #print(f"DEBUG: Table starts at y={table_start_y}")
            break

    # Filter to table area
    table_items = [item for item in ocr_items if item['y'] >= table_start_y]
    #print(f"DEBUG: {len(table_items)} items in table area")

    # Get all unique person names
    all_names = []

    # First pass: collect obvious names
    for item in table_items:
        if is_person_name(item['text']) and not is_company_name(item['text']):
            all_names.append(item)

    # Second pass: be more lenient for bottom area (handwritten)
    max_y = max(item['y'] for item in table_items) if table_items else 0
    bottom_threshold = max_y - 150

    #print(f"DEBUG: Looking for handwritten names in bottom area (y >= {bottom_threshold})")

    for item in table_items:
        if item['y'] >= bottom_threshold:
            text = item['text'].strip()

            if (len(text) >= 2 and
                    text[0].isalpha() and
                    sum(c.isalpha() for c in text) >= 2 and
                    not any(keyword in text.lower() for keyword in
                            ['cash', 'online', 'payment', 'mode', 'date', 'fees', 'company', 'category']) and
                    not re.match(r'^\d+', text)):

                # If not already in all_names
                if text.lower() not in [n['text'].lower() for n in all_names]:
                    all_names.append(item)
                    #print(f"DEBUG: Added handwritten name: '{text}' at y={item['y']:.0f}")

    # Remove duplicates
    def name_similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    unique_names = []
    for item in all_names:
        is_duplicate = False
        for existing in unique_names:
            y_close = abs(item['y'] - existing['y']) < 12
            similarity = name_similarity(item['text'], existing['text'])

            if y_close and similarity > 0.85:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_names.append(item)

    all_names = sorted(unique_names, key=lambda x: x['y'])

    #print(f"DEBUG: Found {len(all_names)} unique person names:")
    for i, name_item in enumerate(all_names):
        print(f"  {i + 1:2d}: '{name_item['text']}' at y={name_item['y']:.0f}")

    # For each name, find associated data in same row
    records = []

    for name_item in all_names:
        name_y = name_item['y']
        name_x = name_item['x']
        name_text = name_item['text']

        # Find items in same row (within 25 pixels vertically)
        row_items = []
        for item in table_items:
            if (abs(item['y'] - name_y) <= 25 and
                    item['x'] > name_x and
                    item['text'] != name_text):
                row_items.append(item)

        # Sort row items by X coordinate
        row_items.sort(key=lambda x: x['x'])

        # Create record
        record = {
            'name': name_text,
            'company_name': '',
            'category': '',
            'invited_by': '',
            'fees': '',
            'payment_mode': '',
            'date': '',
            'note': ''
        }

        # Assign items to columns based on content type and position
        for item in row_items:
            text = item['text'].strip()

            if is_date_format(text) and not record['date']:
                record['date'] = text
            elif is_fees_amount(text) and not record['fees']:
                record['fees'] = text
            elif is_payment_mode(text) and not record['payment_mode']:
                record['payment_mode'] = text
            elif is_company_name(text) and not record['company_name']:
                record['company_name'] = text
            elif is_person_name(text) and not record['invited_by']:
                record['invited_by'] = text
            elif not record['category'] and len(text) >= 3:
                record['category'] = text
            elif not record['note']:
                record['note'] = text

        records.append(record)

    #print(f"DEBUG: Created {len(records)} individual records")

    # Show final records
    #print("\nDEBUG: Final extracted records:")
    for i, record in enumerate(records):
        print(f"  {i + 1:2d}: '{record['name'][:20]:20s}' | Company: '{record['company_name'][:15]:15s}' | "
              f"Cat: '{record['category'][:10]:10s}' | Fees: '{record['fees']:8s}' | "
              f"Mode: '{record['payment_mode']:8s}' | Date: '{record['date'][:10]:10s}'")

    # Convert to DataFrame
    if records:
        df_data = []
        for record in records:
            df_data.append([
                record['name'],
                record['company_name'],
                record['category'],
                record['invited_by'],
                record['fees'],
                record['payment_mode'],
                record['date'],
                record['note']
            ])

        df = pd.DataFrame(df_data, columns=COLUMNS)
    else:
        df = pd.DataFrame([['No valid data found', '', '', '', '', '', '', '']], columns=COLUMNS)

    # Remove any completely empty rows
    df = df[df.apply(lambda x: any(x.astype(str).str.strip() != ''), axis=1)]

    #print(f"DEBUG: Final DataFrame shape: {df.shape}")

    return df


def test_complete_extraction(image_path):
    """Test complete extraction for 8 columns"""
    try:
        #print("=" * 80)
        #print("8-COLUMN EXTRACTION - Name, Company, Category, Invited by, Fees, Payment, Date, Note")
        #print("=" * 80)

        result_df = extract_data_from_image_v2(image_path)

        #print(f"\nFinal Results - {len(result_df)} records:")
        #print(result_df.to_string(index=False, max_colwidth=20))

        return result_df

    except Exception as e:
        print(f"Complete extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    image_path = "accounting.jpeg"

    if os.path.exists(image_path):
        df = test_complete_extraction(image_path)
        if df is not None:
            print(f"\nSuccess! Extracted {len(df)} records.")

            # Save to CSV
            #output_path = "extracted_8columns.csv"
            #df.to_csv(output_path, index=False)
            #print(f"Results saved to: {output_path}")
        else:
            print("Extraction failed.")
    else:
        print(f"Image file not found: {image_path}")