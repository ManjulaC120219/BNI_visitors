import pandas as pd
import numpy as np
import os, re
from collections import defaultdict
from sklearn.cluster import DBSCAN
from datetime import datetime
import warnings
from paddleocr import PaddleOCR
import cv2
import tempfile
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_mkldnn"] = "0"

COLUMNS = ['Name', 'Company Name', 'Invited by', 'Fees', 'Payment Mode', 'Date', 'Note']


def extract_date_from_top_corner(image_path):
    """Extract handwritten date from top-right corner - AGGRESSIVE"""
    try:
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Try multiple regions to find the date
        regions = [
            (0, int(height * 0.15), int(width * 0.6), width),  # Top-right 40%
            (0, int(height * 0.20), int(width * 0.5), width),  # Top-right 50%
            (0, int(height * 0.10), int(width * 0.7), width),  # Top-right 30%
        ]

        all_text_found = []

        for y1, y2, x1, x2 in regions:
            try:
                corner = img[y1:y2, x1:x2]

                # Save temporary crop
                temp_path = tempfile.mktemp(suffix='.jpg')
                cv2.imwrite(temp_path, corner)

                # OCR with handwriting settings
                ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang='en',
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.05,
                    det_db_box_thresh=0.1,
                    det_db_unclip_ratio=4.0
                )

                result = ocr.ocr(temp_path, cls=False)
                os.remove(temp_path)

                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0].strip()
                            all_text_found.append(text)
                            print(f"DEBUG: Found text in corner: '{text}'")
            except:
                continue

        # Look for date patterns in all found text
        for text in all_text_found:
            # Match various date formats
            if re.search(r'\d{1,2}[-/\s]*(?:sep|sept|oct|nov|dec|jan|feb|mar|apr|may|jun|jul|aug)', text.lower()):
                print(f"DEBUG: Detected date from corner: '{text}'")
                return text
            if is_date_format(text):
                print(f"DEBUG: Detected date from corner: '{text}'")
                return text

        print("DEBUG: No date found in corner")
        return None
    except Exception as e:
        print(f"DEBUG: Date extraction from corner failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def cluster_by_rows(items, y_tolerance=30):
    """Cluster OCR items into rows using DBSCAN"""
    if not items:
        return []

    y_coords = np.array([[item['y']] for item in items])
    clustering = DBSCAN(eps=y_tolerance, min_samples=1).fit(y_coords)
    labels = clustering.labels_

    rows = defaultdict(list)
    for item, label in zip(items, labels):
        rows[label].append(item)

    sorted_rows = []
    for label in sorted(rows.keys(), key=lambda l: np.mean([i['y'] for i in rows[l]])):
        row_items = sorted(rows[label], key=lambda x: x['x'])
        sorted_rows.append(row_items)

    return sorted_rows


def detect_column_positions(header_items, all_items):
    """Detect column X positions from header and data"""
    columns = {
        'name': None,
        'company': None,
        'invited': None,
        'fees': None,
        'payment': None,
        'date': None
    }

    # Try to find header positions
    for item in header_items:
        text_lower = item['text'].lower()
        if 'name' in text_lower and 'company' not in text_lower:
            columns['name'] = item['x']
        elif 'company' in text_lower:
            columns['company'] = item['x']
        elif 'invited' in text_lower:
            columns['invited'] = item['x']
        elif 'fee' in text_lower:
            columns['fees'] = item['x']
        elif 'payment' in text_lower or 'mode' in text_lower:
            columns['payment'] = item['x']
        elif 'date' in text_lower:
            columns['date'] = item['x']

    # If header not found, estimate from data distribution
    if not any(columns.values()):
        x_positions = sorted([item['x'] for item in all_items])
        if len(x_positions) >= 6:
            x_array = np.array([[x] for x in x_positions])
            clustering = DBSCAN(eps=50, min_samples=2).fit(x_array)

            unique_labels = sorted(set(clustering.labels_))
            col_positions = []
            for label in unique_labels:
                if label != -1:
                    cluster_points = [x_positions[i] for i, l in enumerate(clustering.labels_) if l == label]
                    col_positions.append(np.mean(cluster_points))

            col_positions = sorted(col_positions)
            if len(col_positions) >= 6:
                columns['name'] = col_positions[0]
                columns['company'] = col_positions[1]
                columns['invited'] = col_positions[2]
                columns['fees'] = col_positions[3]
                columns['payment'] = col_positions[4]
                columns['date'] = col_positions[5]

    print(f"DEBUG: Detected column positions: {columns}")
    return columns


def assign_to_column(item, column_positions, tolerance=80):
    """Assign item to nearest column based on X position"""
    if not any(column_positions.values()):
        return None

    x = item['x']
    closest_col = None
    min_distance = float('inf')

    for col_name, col_x in column_positions.items():
        if col_x is not None:
            distance = abs(x - col_x)
            if distance < min_distance and distance < tolerance:
                min_distance = distance
                closest_col = col_name

    return closest_col


def smart_merge_names_in_row(row_items, column_positions, name_col_tolerance=100):
    """Enhanced name merging with better logic"""
    if not row_items or len(row_items) < 2:
        return row_items

    name_col_x = column_positions.get('name')
    merged = []
    skip_next = set()

    for i, item in enumerate(row_items):
        if i in skip_next:
            continue

        current_item = dict(item)

        # Look ahead for potential name parts
        if i + 1 < len(row_items):
            next_item = row_items[i + 1]

            # Calculate distances
            x_gap = next_item['x'] - (item['x'] + item['width'])
            y_gap = abs(item['y'] - next_item['y'])

            should_merge = False

            # STRICT CRITERIA for merging names
            if name_col_x:
                # Both items must be in name column area
                item_in_name_col = abs(item['x'] - name_col_x) < name_col_tolerance
                next_in_name_col = abs(next_item['x'] - name_col_x) < name_col_tolerance + 50

                if item_in_name_col and next_in_name_col:
                    # Very strict distance requirements
                    if x_gap < 20 and y_gap < 8:
                        # Both must look like valid name parts
                        if is_valid_name(item['text']) and is_valid_name(next_item['text']):
                            # Additional check: neither should contain numbers
                            if not any(c.isdigit() for c in item['text'] + next_item['text']):
                                should_merge = True

            # Fallback: merge very close text that looks like split names
            if not should_merge:
                if x_gap < 15 and y_gap < 5:
                    # Check text similarity (could be OCR reading same text twice)
                    similarity = SequenceMatcher(None,
                                                 item['text'].lower(),
                                                 next_item['text'].lower()).ratio()

                    # If not similar, check if both are valid name parts
                    if similarity < 0.5:  # Not duplicates
                        both_valid = (is_valid_name(item['text']) and
                                      is_valid_name(next_item['text']))
                        both_short = len(item['text']) <= 4 or len(next_item['text']) <= 4
                        no_numbers = not any(c.isdigit() for c in item['text'] + next_item['text'])

                        if both_valid and both_short and no_numbers:
                            should_merge = True

            if should_merge:
                current_item['text'] = f"{item['text']} {next_item['text']}"
                current_item['width'] = (next_item['x'] + next_item['width']) - item['x']
                skip_next.add(i + 1)
                print(f"DEBUG: Merged name parts: '{item['text']}' + '{next_item['text']}' = '{current_item['text']}'")

        merged.append(current_item)

    return merged


def smart_merge_names_improved(row_items, column_positions, name_col_tolerance=100):
    """Enhanced name merging with better logic"""
    if not row_items or len(row_items) < 2:
        return row_items

    name_col_x = column_positions.get('name')
    merged = []
    skip_next = set()

    for i, item in enumerate(row_items):
        if i in skip_next:
            continue

        current_item = dict(item)

        # Look ahead for potential name parts
        if i + 1 < len(row_items):
            next_item = row_items[i + 1]

            # Calculate distances
            x_gap = next_item['x'] - (item['x'] + item['width'])
            y_gap = abs(item['y'] - next_item['y'])

            should_merge = False

            # STRICT CRITERIA for merging names
            if name_col_x:
                # Both items must be in name column area
                item_in_name_col = abs(item['x'] - name_col_x) < name_col_tolerance
                next_in_name_col = abs(next_item['x'] - name_col_x) < name_col_tolerance + 50

                if item_in_name_col and next_in_name_col:
                    # Very strict distance requirements
                    if x_gap < 20 and y_gap < 8:
                        # Both must look like valid name parts
                        if is_valid_name(item['text']) and is_valid_name(next_item['text']):
                            # Additional check: neither should contain numbers
                            if not any(c.isdigit() for c in item['text'] + next_item['text']):
                                should_merge = True

            # Fallback: merge very close text that looks like split names
            if not should_merge:
                if x_gap < 15 and y_gap < 5:
                    # Check text similarity (could be OCR reading same text twice)
                    similarity = SequenceMatcher(None,
                                                 item['text'].lower(),
                                                 next_item['text'].lower()).ratio()

                    # If not similar, check if both are valid name parts
                    if similarity < 0.5:  # Not duplicates
                        both_valid = (is_valid_name(item['text']) and
                                      is_valid_name(next_item['text']))
                        both_short = len(item['text']) <= 4 or len(next_item['text']) <= 4
                        no_numbers = not any(c.isdigit() for c in item['text'] + next_item['text'])

                        if both_valid and both_short and no_numbers:
                            should_merge = True

            if should_merge:
                current_item['text'] = f"{item['text']} {next_item['text']}"
                current_item['width'] = (next_item['x'] + next_item['width']) - item['x']
                skip_next.add(i + 1)
                print(f"DEBUG: Merged name parts: '{item['text']}' + '{next_item['text']}' = '{current_item['text']}'")

        merged.append(current_item)

    return merged

def is_valid_name(text):
    """Enhanced name validation with stricter rules"""
    text = text.strip()

    # Minimum length check
    if len(text) < 2:
        return False

    # Reject pure numbers or symbols
    if re.match(r'^[\d\s\.\,\-\/\:\(\)\[\]]+$', text):
        return False

    # Reject common non-name patterns
    reject_patterns = [
        r'^\d+[-/]\d+',  # Dates like 12/11
        r'^[=\-_]+$',  # Only special chars
        r'^\d+\s*/-',  # Fee patterns
    ]
    for pattern in reject_patterns:
        if re.search(pattern, text):
            return False

    # Reject headers and common keywords
    headers = [
        'name', 'company', 'invited', 'fees', 'payment', 'date', 'note',
        'signature', 'mode', 'visitor', 'sheet', 'brilliance', 'bni',
        'total', 'amount', 'cash', 'online', 'upi', 'card', 'cheque',
        'serial', 'no.', 'sr.'
    ]
    text_lower = text.lower()
    if text_lower in headers or any(h == text_lower for h in headers):
        return False

    # Reject company-specific keywords
    company_keywords = [
        'ltd', 'inc', 'corp', 'pvt', 'llc', 'llp', 'industries',
        'technologies', 'jewellery', 'infotech', 'services'
    ]
    if any(keyword in text_lower for keyword in company_keywords):
        return False

    # Must have at least 2 alphabetic characters
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < 2:
        return False

    # Ratio check: at least 60% should be letters
    if len(text) > 0:
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < 0.6:
            return False

    # Check for common name patterns (optional but helpful)
    # Names typically have:
    # - Mix of upper and lowercase OR all caps
    # - May contain spaces, hyphens, apostrophes
    # - Shouldn't start with numbers
    if text[0].isdigit():
        return False

    # Reject if it looks like merged data (name=value pattern)
    if '=' in text or text.count('=') > 0:
        # Check if it's not just a name with = at the end
        if re.search(r'[A-Za-z]+=\d', text):
            return False

    return True


def is_company_name(text):
    """Detect company names"""
    text = text.strip()

    if len(text) < 2:
        return False

    company_keywords = ['ltd', 'inc', 'corp', 'pvt', 'llc', 'llp', 'co.', 'industries', 'technologies',
                        'group', 'jewellery', 'jewel', 'wedding', 'infotech', 'services', 'land', 'intucent',
                        'wedingrings', 'fewelly', 'ictoa']
    lower = text.lower()

    if any(keyword in lower for keyword in company_keywords):
        return True

    return False


def extract_name_from_row(row_items, column_positions, name_col_tolerance=80):
    """Extract name with improved logic"""
    name_col_x = column_positions.get('name')

    # Priority 1: Items in name column position
    if name_col_x:
        name_candidates = []
        for item in row_items:
            distance = abs(item['x'] - name_col_x)
            if distance < name_col_tolerance:
                if is_valid_name(item['text']):
                    name_candidates.append((item, distance))

        if name_candidates:
            # Get closest item to name column
            name_candidates.sort(key=lambda x: x[1])
            best_name = name_candidates[0][0]['text']

            # Check if next item should be merged
            best_idx = row_items.index(name_candidates[0][0])
            if best_idx + 1 < len(row_items):
                next_item = row_items[best_idx + 1]
                x_gap = next_item['x'] - (name_candidates[0][0]['x'] + name_candidates[0][0]['width'])
                y_gap = abs(name_candidates[0][0]['y'] - next_item['y'])

                if (x_gap < 20 and y_gap < 8 and
                        is_valid_name(next_item['text']) and
                        abs(next_item['x'] - name_col_x) < name_col_tolerance + 50):
                    best_name = f"{best_name} {next_item['text']}"

            return best_name

    # Priority 2: First valid name in row (leftmost)
    for item in sorted(row_items, key=lambda x: x['x']):
        text = item['text'].strip()

        # Skip serial numbers
        if re.match(r'^[0-9]{1,2}\.?$', text):
            continue

        # Check if it's a valid name
        if is_valid_name(text):
            # Make sure it's not a fee or other data
            if not extract_fees(text):  # Use your existing extract_fees function
                return text

    return ''


def extract_fees(text):
    """Your existing extract_fees function - included for reference"""
    text = text.strip()

    merged_pattern = r'[A-Za-z]+[=\s]*(\d{3,})\s*[-/]*'
    merged_match = re.search(merged_pattern, text)
    if merged_match:
        try:
            amount = int(merged_match.group(1))
            if 100 <= amount < 100000:
                return str(amount)
        except:
            pass

    clean = text.replace('₹', '').replace('Rs', '').replace('rs', '').replace(',', '').strip()

    patterns = [
        r'(\d+)\s*/-',
        r'=\s*(\d+)',
        r'^(\d{3,})$',
        r'(\d{3,})',
    ]

    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            try:
                amount = int(match.group(1))
                if 100 <= amount < 100000:
                    return amount
            except:
                pass

    return ''
def extract_name_from_merged(text):
    """Extract name from merged text like 'Khalid=1400/-'"""
    match = re.match(r'^([A-Za-z\s]+)(?=[=\d])', text)
    if match:
        return match.group(1).strip()
    return ''


def is_payment_mode(text):
    """Detect payment mode"""
    text = text.lower().strip()
    modes = ['cash', 'online', 'upi', 'card', 'cheque', 'bank', 'transfer']
    return any(mode == text or mode in text for mode in modes)


def is_date_format(text):
    """Detect date patterns"""
    patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{1,2}\s*[-]\s*[A-Za-z]{3,9}',
        r'[A-Za-z]{3,9}\s*[-]\s*\d{1,2}',
        r'\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}',
        r'\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4}'
    ]
    return any(re.search(pattern, text.strip()) for pattern in patterns)


def is_signature_or_check(text):
    """Identify signature or checkmark symbols"""
    text = text.strip()
    if text in ['✓', '√', '×', '✗', 'V', 'v', 'x', 'X', '/', '\\']:
        return True
    if len(text) <= 2 and not text.isalnum():
        return True
    return False


def merge_similar_text(items, similarity_threshold=0.75, x_tolerance=40):
    """Merge OCR items that are likely the same text"""
    if not items:
        return []

    merged = []
    used = set()

    for i, item1 in enumerate(items):
        if i in used:
            continue

        similar_group = [item1]
        for j, item2 in enumerate(items):
            if j <= i or j in used:
                continue

            x_close = abs(item1['x'] - item2['x']) < x_tolerance
            y_close = abs(item1['y'] - item2['y']) < 8

            if x_close and y_close:
                similarity = SequenceMatcher(None, item1['text'].lower(), item2['text'].lower()).ratio()
                if similarity > similarity_threshold:
                    similar_group.append(item2)
                    used.add(j)

        best_item = max(similar_group, key=lambda x: x['score'])
        merged.append(best_item)
        used.add(i)

    return merged


def get_image_dimensions(image_path):
    """Get image dimensions"""
    img = cv2.imread(image_path)
    if img is not None:
        height, width = img.shape[:2]
        return width, height
    return None, None


def extract_data_from_image_v2(image_path):
    """Improved extraction with date from corner"""

    try:
        # Extract date from top-right corner first
        corner_date = extract_date_from_top_corner(image_path)
        print(f"DEBUG: Extracted corner date: {corner_date}")

        # Get image dimensions for signature exclusion
        img_width, img_height = get_image_dimensions(image_path)

        # Primary OCR with balanced settings for printed text
        ocr_printed = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6
        )

        result_printed = ocr_printed.ocr(image_path, cls=True)

        # Secondary OCR with settings optimized for handwritten text
        ocr_handwritten = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.08,
            det_db_box_thresh=0.15,
            det_db_unclip_ratio=3.0
        )

        result_handwritten = ocr_handwritten.ocr(image_path, cls=True)

        # Combine results
        all_items = []
        if result_printed and result_printed[0]:
            all_items.extend(result_printed[0])
        if result_handwritten and result_handwritten[0]:
            all_items.extend(result_handwritten[0])

        if not all_items:
            raise RuntimeError("No OCR results")

        # Parse OCR items
        ocr_items = []
        header_items = []
        signature_x_threshold = img_width * 0.85 if img_width else 999999

        for line in all_items:
            if line and len(line) >= 2:
                poly, (text, score) = line[0], line[1]

                if score < 0.15:
                    continue

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                # Skip signature column
                if center_x > signature_x_threshold:
                    continue

                if is_signature_or_check(text.strip()):
                    continue

                item = {
                    'text': text.strip(),
                    'score': float(score),
                    'x': center_x,
                    'y': center_y,
                    'width': max(xs) - min(xs),
                    'height': max(ys) - min(ys)
                }

                ocr_items.append(item)

                text_lower = text.lower()
                if any(h in text_lower for h in ['name', 'company', 'invited', 'fees', 'payment', 'date', 'mode']):
                    header_items.append(item)

        print(f"DEBUG: Total OCR items: {len(ocr_items)}")

        # Merge similar overlapping text
        ocr_items = merge_similar_text(ocr_items, similarity_threshold=0.75, x_tolerance=40)
        print(f"DEBUG: After merging similar: {len(ocr_items)}")

        # Detect column positions
        column_positions = detect_column_positions(header_items, ocr_items)

        # Find table start
        table_start_y = 0
        for item in header_items:
            table_start_y = max(table_start_y, item['y'] + item['height'] + 10)

        if table_start_y == 0:
            if ocr_items:
                sorted_by_y = sorted(ocr_items, key=lambda x: x['y'])
                if len(sorted_by_y) > 5:
                    table_start_y = sorted_by_y[4]['y']

        # Filter to table area
        table_items = [item for item in ocr_items if item['y'] > table_start_y]
        print(f"DEBUG: {len(table_items)} items in table area")

        # Cluster items into rows
        rows = cluster_by_rows(table_items, y_tolerance=30)
        print(f"DEBUG: Detected {len(rows)} rows")

        # Extract data from EVERY row
        records = []

        for row_idx, row_items in enumerate(rows):
            if len(row_items) < 1:
                continue

            # Smart merge of split names
            row_items = smart_merge_names_in_row(row_items, column_positions)
            row_items = sorted(row_items, key=lambda x: x['x'])

            row_text = [f"{item['text']}" for item in row_items]
            print(f"\nRow {row_idx + 1}: {row_text}")

            record = {
                'name': '',
                'company_name': '',
                'invited_by': '',
                'fees': '',
                'payment_mode': '',
                'date': corner_date if corner_date else '',  # Use corner date
                'note': ''
            }

            column_data = {
                'name': [],
                'company': [],
                'invited': [],
                'fees': [],
                'payment': [],
                'date': [],
                'other': []
            }

            for item in row_items:
                text = item['text'].strip()

                # Skip serial numbers
                if re.match(r'^[0-9]{1,2}\.?$', text) and item['x'] < 100:
                    continue

                if len(text) <= 1:
                    continue

                # Extract fees (handles merged text)
                fees = extract_fees(text)
                if fees:
                    column_data['fees'].append(fees)
                    name_part = extract_name_from_merged(text)
                    if name_part and is_valid_name(name_part):
                        column_data['name'].append(name_part)
                    continue

                # Check for date (but corner date takes priority)
                if is_date_format(text) and not corner_date:
                    column_data['date'].append(text)
                    continue

                # Check for payment mode
                if is_payment_mode(text):
                    column_data['payment'].append(text)
                    continue

                # Assign to column
                col = assign_to_column(item, column_positions)

                if col:
                    column_data[col].append(text)
                else:
                    if is_company_name(text):
                        column_data['company'].append(text)
                    elif is_valid_name(text):
                        if not column_data['name']:
                            column_data['name'].append(text)
                        elif not column_data['invited']:
                            column_data['invited'].append(text)
                        else:
                            column_data['other'].append(text)
                    else:
                        column_data['other'].append(text)

            # Populate record
            # In your row processing loop, replace the name extraction with:
            record['name'] = extract_name_from_row(row_items, column_positions, name_col_tolerance=80)
            record['company_name'] = ' '.join(column_data['company']) if column_data['company'] else ''
            record['invited_by'] = ' '.join(column_data['invited']) if column_data['invited'] else ''
            record['fees'] = column_data['fees'][0] if column_data['fees'] else ''
            record['payment_mode'] = column_data['payment'][0] if column_data['payment'] else ''
            # Only override with row date if no corner date
            if not record['date'] and column_data['date']:
                record['date'] = column_data['date'][0]
            record['note'] = ' '.join(column_data['other']) if column_data['other'] else ''

            # Fallback name extraction
            if not record['name'] and record['fees']:
                for item in row_items:
                    name_part = extract_name_from_merged(item['text'])
                    if name_part:
                        record['name'] = name_part
                        break

            if not record['name']:
                for item in row_items:
                    if is_valid_name(item['text']) and len(item['text']) >= 2:
                        record['name'] = item['text']
                        break

            # Add ALL rows with any meaningful data
            has_data = any([
                record['name'],
                record['company_name'],
                record['fees'],
                len(record['note']) > 2
            ])

            if has_data:
                records.append(record)
                print(f"  ✓ Extracted: Name='{record['name']}', Company='{record['company_name']}', "
                      f"Invited='{record['invited_by']}', Fees='{record['fees']}', "
                      f"Payment='{record['payment_mode']}', Date='{record['date']}'")
            else:
                if row_text:
                    record['note'] = ' '.join(row_text)
                    records.append(record)
                    print(f"  ⚠ Extracted with minimal data: {record['note']}")

        print(f"\nDEBUG: Extracted {len(records)} valid records")

        if records:
            df_data = [[r['name'], r['company_name'], r['invited_by'],
                        r['fees'], r['payment_mode'], r['date'], r['note']]
                       for r in records]
            df = pd.DataFrame(df_data, columns=COLUMNS)
        else:
            df = pd.DataFrame(columns=COLUMNS)

        return df

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=COLUMNS)


def test_extraction(image_path):
    """Test the improved extraction"""
    try:
        print("=" * 80)
        print("IMPROVED OCR EXTRACTION (With Date from Corner)")
        print("=" * 80)

        df = extract_data_from_image_v2(image_path)

        print(f"\n{'=' * 80}")
        print(f"FINAL RESULTS: {len(df)} records extracted")
        print("=" * 80)
        if len(df) > 0:
            print(df.to_string(index=False))
        else:
            print("No records extracted")

        return df

    except Exception as e:
        print(f"Extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    image_path = "accounting.jpeg"

    if os.path.exists(image_path):
        df = test_extraction(image_path)
        if df is not None and len(df) > 0:
            print(f"\n✓ Success! Extracted {len(df)} records.")
        else:
            print("✗ Extraction failed or no records found.")
    else:
        print(f"✗ Image file not found: {image_path}")