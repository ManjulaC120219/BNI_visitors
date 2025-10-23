import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from sklearn.cluster import DBSCAN
from paddleocr import PaddleOCR
import cv2
from difflib import SequenceMatcher
import warnings

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_mkldnn"] = "0"

COLUMNS = ['Name', 'Company Name', 'Invited by', 'Fees', 'Payment Mode', 'Note']


def detect_document_type(image_path):
    """
    Detect if document is handwritten or printed
    Returns: 'handwritten' or 'printed'
    """
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        result = ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return 'printed'

        scores = []
        text_heights = []
        char_widths = []

        for line in result[0]:
            if line and len(line) >= 2:
                poly, (text, score) = line[0], line[1]
                scores.append(score)

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                text_heights.append(max(ys) - min(ys))

                # Calculate character width consistency
                width = max(xs) - min(xs)
                if len(text) > 0:
                    char_widths.append(width / len(text))

        if not scores:
            return 'printed'

        avg_score = np.mean(scores)
        height_variance = np.var(text_heights) if len(text_heights) > 1 else 0
        width_variance = np.var(char_widths) if len(char_widths) > 1 else 0

        # Enhanced criteria for printed documents:
        # 1. Very high confidence (>0.92) AND low height variance (<15)
        # 2. OR high confidence (>0.88) AND very low height variance (<10)
        # 3. OR consistent character widths (low variance <50)

        is_printed = False

        # Criteria 1: Very high confidence + consistent heights
        if avg_score > 0.92 and height_variance < 15:
            is_printed = True

        # Criteria 2: High confidence + very consistent heights
        elif avg_score > 0.88 and height_variance < 10:
            is_printed = True

        # Criteria 3: Consistent character widths (typical of printed text)
        elif width_variance < 50 and avg_score > 0.85:
            is_printed = True

        doc_type = 'printed' if is_printed else 'handwritten'

        print(f"\n{'=' * 80}")
        print(f"DOCUMENT TYPE DETECTED: {doc_type.upper()}")
        print(f"  Average OCR confidence: {avg_score:.3f}")
        print(f"  Height variance: {height_variance:.2f}")
        print(f"  Width variance: {width_variance:.2f}")
        print(f"  Decision: ", end="")
        if is_printed:
            if avg_score > 0.92 and height_variance < 15:
                print("High confidence + consistent heights")
            elif avg_score > 0.88 and height_variance < 10:
                print("Very consistent heights")
            else:
                print("Consistent character widths")
        else:
            print("Characteristics indicate handwritten")
        print(f"{'=' * 80}\n")

        return doc_type

    except Exception as e:
        print(f"Document detection failed: {e}, defaulting to printed")
        return 'printed'


# ============================================================================
# CODE1.PY LOGIC - FOR HANDWRITTEN DOCUMENTS
# ============================================================================

def cluster_by_rows_handwritten(items, y_tolerance=30):
    """Cluster OCR items into rows using DBSCAN (Code1 logic)"""
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


def extract_handwritten(image_path):
    """COMPLETE CODE1.PY EXTRACTION LOGIC"""
    try:
        img_width, img_height = get_image_dimensions(image_path)

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

        all_items = []
        if result_printed and result_printed[0]:
            all_items.extend(result_printed[0])
        if result_handwritten and result_handwritten[0]:
            all_items.extend(result_handwritten[0])

        if not all_items:
            raise RuntimeError("No OCR results")

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
                if any(h in text_lower for h in ['name', 'company', 'invited', 'fees', 'payment', 'notes']):
                    header_items.append(item)

        print(f"DEBUG: Total OCR items: {len(ocr_items)}")

        ocr_items = merge_similar_text(ocr_items, similarity_threshold=0.75, x_tolerance=40)
        print(f"DEBUG: After merging similar: {len(ocr_items)}")

        column_positions = detect_column_positions(header_items, ocr_items)

        table_start_y = 0
        for item in header_items:
            table_start_y = max(table_start_y, item['y'] + item['height'] + 10)

        if table_start_y == 0:
            if ocr_items:
                sorted_by_y = sorted(ocr_items, key=lambda x: x['y'])
                if len(sorted_by_y) > 5:
                    table_start_y = sorted_by_y[4]['y']

        table_items = [item for item in ocr_items if item['y'] > table_start_y]
        print(f"DEBUG: {len(table_items)} items in table area")

        rows = cluster_by_rows_handwritten(table_items, y_tolerance=30)
        print(f"DEBUG: Detected {len(rows)} rows")

        records = []

        for row_idx, row_items in enumerate(rows):
            if len(row_items) < 1:
                continue

            row_items = smart_merge_names_handwritten(row_items, column_positions)
            row_items = sorted(row_items, key=lambda x: x['x'])

            row_text = [f"{item['text']}" for item in row_items]
            print(f"\nRow {row_idx + 1}: {row_text}")

            record = {
                'name': '',
                'company_name': '',
                'invited_by': '',
                'fees': '',
                'payment_mode': '',
                'note': ''
            }

            column_data = {
                'name': [],
                'company': [],
                'invited': [],
                'fees': [],
                'payment': [],
                'notes': []
            }

            for item in row_items:
                text = item['text'].strip()

                if re.match(r'^[0-9]{1,2}\.?$', text) and item['x'] < 100:
                    continue

                if len(text) <= 1:
                    continue

                fees = extract_fees(text)
                if fees:
                    column_data['fees'].append(fees)
                    name_part = extract_name_from_merged(text)
                    if name_part and is_valid_name(name_part):
                        column_data['name'].append(name_part)
                    continue

                if is_payment_mode(text):
                    column_data['payment'].append(text)
                    continue

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

            record['name'] = extract_name_from_row_handwritten(row_items, column_positions, name_col_tolerance=80)
            record['company_name'] = ' '.join(column_data['company']) if column_data['company'] else ''
            record['invited_by'] = ' '.join(column_data['invited']) if column_data['invited'] else ''
            record['fees'] = column_data['fees'][0] if column_data['fees'] else ''
            record['payment_mode'] = column_data['payment'][0] if column_data['payment'] else ''

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
                      f"Payment='{record['payment_mode']}'")
            else:
                if row_text:
                    record['note'] = ' '.join(row_text)
                    records.append(record)
                    print(f"  ⚠ Extracted with minimal data: {record['note']}")

        print(f"\nDEBUG: Extracted {len(records)} valid records")

        if records:
            df_data = [[r['name'], r['company_name'], r['invited_by'],
                        r['fees'], r['payment_mode'], r['note']]
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


def smart_merge_names_handwritten(row_items, column_positions, name_col_tolerance=100):
    """Code1 name merging logic"""
    if not row_items or len(row_items) < 2:
        return row_items

    name_col_x = column_positions.get('name')
    merged = []
    skip_next = set()

    for i, item in enumerate(row_items):
        if i in skip_next:
            continue

        current_item = dict(item)

        if i + 1 < len(row_items):
            next_item = row_items[i + 1]
            x_gap = next_item['x'] - (item['x'] + item['width'])
            y_gap = abs(item['y'] - next_item['y'])
            should_merge = False

            if name_col_x:
                item_in_name_col = abs(item['x'] - name_col_x) < name_col_tolerance
                next_in_name_col = abs(next_item['x'] - name_col_x) < name_col_tolerance + 50

                if item_in_name_col and next_in_name_col:
                    if x_gap < 20 and y_gap < 8:
                        if is_valid_name(item['text']) and is_valid_name(next_item['text']):
                            if not any(c.isdigit() for c in item['text'] + next_item['text']):
                                should_merge = True

            if not should_merge:
                if x_gap < 15 and y_gap < 5:
                    similarity = SequenceMatcher(None,
                                                 item['text'].lower(),
                                                 next_item['text'].lower()).ratio()

                    if similarity < 0.5:
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

        merged.append(current_item)

    return merged


def extract_name_from_row_handwritten(row_items, column_positions, name_col_tolerance=80):
    """Code1 name extraction logic"""
    name_col_x = column_positions.get('name')

    if name_col_x:
        name_candidates = []
        for item in row_items:
            distance = abs(item['x'] - name_col_x)
            if distance < name_col_tolerance:
                if is_valid_name(item['text']):
                    name_candidates.append((item, distance))

        if name_candidates:
            name_candidates.sort(key=lambda x: x[1])
            best_name = name_candidates[0][0]['text']

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

    for item in sorted(row_items, key=lambda x: x['x']):
        text = item['text'].strip()

        if re.match(r'^[0-9]{1,2}\.?$', text):
            continue

        if is_valid_name(text):
            if not extract_fees(text):
                return text

    return ''


# ============================================================================
# CODE2.PY LOGIC - FOR PRINTED DOCUMENTS
# ============================================================================

def cluster_by_rows_printed(items, y_tolerance=20):
    """Code2 clustering with adaptive tolerance"""
    if not items:
        return []

    y_coords = np.array([[item['y']] for item in items])
    y_values = sorted([item['y'] for item in items])

    if len(y_values) > 1:
        gaps = [y_values[i + 1] - y_values[i] for i in range(len(y_values) - 1) if y_values[i + 1] - y_values[i] > 1]
        if gaps:
            median_gap = np.median(gaps)
            adaptive_tolerance = max(15, min(median_gap * 0.7, 30))
            print(f"DEBUG: Adaptive y_tolerance = {adaptive_tolerance:.1f} (median gap: {median_gap:.1f})")
        else:
            adaptive_tolerance = y_tolerance
    else:
        adaptive_tolerance = y_tolerance

    clustering = DBSCAN(eps=adaptive_tolerance, min_samples=1).fit(y_coords)
    labels = clustering.labels_

    rows = defaultdict(list)
    for item, label in zip(items, labels):
        rows[label].append(item)

    print(f"DEBUG: Clustered into {len(rows)} rows with tolerance {adaptive_tolerance:.1f}")

    sorted_rows = []
    for label in sorted(rows.keys(), key=lambda l: np.mean([i['y'] for i in rows[l]])):
        row_items = sorted(rows[label], key=lambda x: x['x'])
        sorted_rows.append(row_items)

    return sorted_rows


def extract_printed(image_path):
    """COMPLETE CODE2.PY EXTRACTION LOGIC"""
    try:
        img_width, img_height = get_image_dimensions(image_path)

        ocr_printed = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.2,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0
        )

        result_printed = ocr_printed.ocr(image_path, cls=True)

        ocr_handwritten = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.03,
            det_db_box_thresh=0.05,
            det_db_unclip_ratio=4.0,
            rec_batch_num=1
        )

        result_handwritten = ocr_handwritten.ocr(image_path, cls=True)

        all_items = []
        if result_printed and result_printed[0]:
            all_items.extend(result_printed[0])
        if result_handwritten and result_handwritten[0]:
            all_items.extend(result_handwritten[0])

        if not all_items:
            raise RuntimeError("No OCR results")

        ocr_items = []
        header_items = []
        signature_x_threshold = img_width * 0.82 if img_width else 999999

        for line in all_items:
            if line and len(line) >= 2:
                poly, (text, score) = line[0], line[1]

                if score < 0.05:
                    continue

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

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
                if any(h in text_lower for h in ['name', 'company', 'invited', 'fees', 'payment', 'notes']):
                    header_items.append(item)

        print(f"DEBUG: Total OCR items: {len(ocr_items)}")

        ocr_items = merge_similar_text(ocr_items, similarity_threshold=0.70, x_tolerance=30)
        print(f"DEBUG: After merging similar: {len(ocr_items)}")

        column_positions = detect_column_positions(header_items, ocr_items)

        table_start_y = 0
        if header_items:
            header_y_positions = [item['y'] for item in header_items]
            max_header_y = max(header_y_positions)

            header_row_items = [item for item in header_items if abs(item['y'] - max_header_y) < 15]

            if header_row_items:
                max_header_bottom = max(item['y'] + item['height'] for item in header_row_items)

                sorted_items = sorted(ocr_items, key=lambda x: x['y'])
                items_after_header = [item for item in sorted_items if item['y'] > max_header_bottom]

                name_col_x = column_positions.get('name', 434.0)

                print(
                    f"DEBUG: Looking for names near X={name_col_x}, between Y={max_header_bottom} and Y={max_header_bottom + 50}")

                potential_names_near_header = []
                for item in items_after_header:
                    if item['y'] < max_header_bottom + 50:
                        x_dist = abs(item['x'] - name_col_x)
                        is_name = is_valid_name(item['text'])
                        is_company = is_company_name(item['text'])

                        if (is_name and len(item['text']) > 5 and x_dist < 150 and not is_company):
                            potential_names_near_header.append(item)

                if potential_names_near_header:
                    first_name_y = min(item['y'] for item in potential_names_near_header)
                    table_start_y = first_name_y - 2
                    first_name_text = [i['text'] for i in potential_names_near_header if i['y'] == first_name_y][0]
                    print(f"DEBUG: Found name '{first_name_text}' at Y={first_name_y:.1f}")
                else:
                    table_start_y = max_header_bottom + 1
                    print(f"DEBUG: No names found near header within 50px")

                print(f"DEBUG: Header bottom: {max_header_bottom:.1f}, Table starts at y={table_start_y}")

        if table_start_y == 0:
            if ocr_items:
                sorted_by_y = sorted(ocr_items, key=lambda x: x['y'])
                potential_headers = [item for item in sorted_by_y if any(
                    h in item['text'].lower() for h in ['name', 'company', 'invited', 'fees', 'payment'])]

                if potential_headers:
                    max_header_y = max(item['y'] + item['height'] for item in potential_headers)
                    table_start_y = max_header_y - 20
                else:
                    skip_count = max(1, int(len(sorted_by_y) * 0.08))
                    table_start_y = sorted_by_y[skip_count]['y'] - 10

        table_items = [item for item in ocr_items if item['y'] > (table_start_y - 30)]

        print(f"DEBUG: {len(table_items)} items in table area (start_y: {table_start_y})")

        rows = cluster_by_rows_printed(table_items, y_tolerance=20)
        print(f"DEBUG: Detected {len(rows)} rows")

        records = []

        for row_idx, row_items in enumerate(rows):
            if len(row_items) < 1:
                continue

            if len(row_items) == 1 and re.match(r'^[0-9]{1,2}\.?$', row_items[0]['text'].strip()):
                continue

            row_items = smart_merge_names_printed(row_items, column_positions)
            row_items = sorted(row_items, key=lambda x: x['x'])

            row_text = [item['text'] for item in row_items]
            print(f"\nRow {row_idx + 1}: {row_text}")

            record = {
                'name': '',
                'company_name': '',
                'invited_by': '',
                'fees': '',
                'payment_mode': '',
                'note': ''
            }

            column_data = {
                'name': [],
                'company': [],
                'invited': [],
                'fees': [],
                'payment': [],
                'notes': []
            }

            for item in row_items:
                text = item['text'].strip()

                if re.match(r'^[0-9]{1,2}\.?$', text) and item['x'] < 150:
                    continue

                if len(text) <= 1:
                    continue

                fees = extract_fees(text)
                if fees:
                    column_data['fees'].append(fees)
                    name_part = extract_name_from_merged(text)
                    if name_part and is_valid_name(name_part):
                        column_data['name'].append(name_part)
                    continue

                if is_payment_mode(text):
                    column_data['payment'].append(text)
                    continue

                col = assign_to_column_printed(item, column_positions, tolerance=90)

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
                            column_data['notes'].append(text)

            record['name'] = extract_name_from_row_printed(row_items, column_positions, name_col_tolerance=120)
            record['company_name'] = ' '.join(column_data['company']) if column_data['company'] else ''
            record['invited_by'] = ' '.join(column_data['invited']) if column_data['invited'] else ''
            record['fees'] = column_data['fees'][0] if column_data['fees'] else ''
            record['payment_mode'] = column_data['payment'][0] if column_data['payment'] else ''
            record['note'] = ' '.join(column_data['notes']) if column_data['notes'] else ''

            if not record['name'] and record['fees']:
                for item in row_items:
                    name_part = extract_name_from_merged(item['text'])
                    if name_part:
                        record['name'] = name_part
                        break

            if not record['name']:
                for item in row_items:
                    if is_valid_name(item['text']) and len(item['text']) >= 3:
                        record['name'] = item['text']
                        break

            has_data = any([record['name'], record['company_name'], record['fees']])

            if has_data:
                records.append(record)
                print(
                    f"  ✓ Extracted: Name='{record['name']}', Company='{record['company_name']}', Fees='{record['fees']}'")

        print(f"\nDEBUG: Extracted {len(records)} valid records")

        if records:
            df_data = [[r['name'], r['company_name'], r['invited_by'], r['fees'], r['payment_mode'], r['note']] for r in
                       records]
            df = pd.DataFrame(df_data, columns=COLUMNS)
        else:
            df = pd.DataFrame(columns=COLUMNS)

        return df

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=COLUMNS)


def smart_merge_names_printed(row_items, column_positions, name_col_tolerance=100):
    """Code2 name merging logic"""
    if not row_items or len(row_items) < 2:
        return row_items

    name_col_x = column_positions.get('name')
    merged = []
    skip_next = set()

    for i, item in enumerate(row_items):
        if i in skip_next:
            continue

        current_item = dict(item)

        if i + 1 < len(row_items):
            next_item = row_items[i + 1]
            x_gap = next_item['x'] - (item['x'] + item['width'])
            y_gap = abs(item['y'] - next_item['y'])
            should_merge = False

            if name_col_x:
                item_in_name_col = abs(item['x'] - name_col_x) < name_col_tolerance
                next_in_name_col = abs(next_item['x'] - name_col_x) < name_col_tolerance + 80

                if item_in_name_col and next_in_name_col:
                    if x_gap < 40 and y_gap < 12:
                        if is_valid_name(item['text']) and is_valid_name(next_item['text']):
                            if not any(c.isdigit() for c in item['text'] + next_item['text']):
                                should_merge = True

            if not should_merge:
                if x_gap < 25 and y_gap < 10:
                    similarity = SequenceMatcher(None, item['text'].lower(), next_item['text'].lower()).ratio()
                    if similarity < 0.5:
                        both_valid = is_valid_name(item['text']) and is_valid_name(next_item['text'])
                        both_short = len(item['text']) <= 6 or len(next_item['text']) <= 6
                        no_numbers = not any(c.isdigit() for c in item['text'] + next_item['text'])
                        if both_valid and both_short and no_numbers:
                            should_merge = True

            if should_merge:
                current_item['text'] = f"{item['text']} {next_item['text']}"
                current_item['width'] = (next_item['x'] + next_item['width']) - item['x']
                skip_next.add(i + 1)

        merged.append(current_item)

    return merged


def extract_name_from_row_printed(row_items, column_positions, name_col_tolerance=120):
    """Code2 name extraction logic"""
    name_col_x = column_positions.get('name')

    if name_col_x:
        name_candidates = []
        for item in row_items:
            distance = abs(item['x'] - name_col_x)
            if distance < name_col_tolerance:
                if is_valid_name(item['text']):
                    name_candidates.append((item, distance))

        if name_candidates:
            name_candidates.sort(key=lambda x: x[1])
            best_name = name_candidates[0][0]['text']
            best_idx = row_items.index(name_candidates[0][0])

            if best_idx + 1 < len(row_items):
                next_item = row_items[best_idx + 1]
                x_gap = next_item['x'] - (name_candidates[0][0]['x'] + name_candidates[0][0]['width'])
                y_gap = abs(name_candidates[0][0]['y'] - next_item['y'])

                if x_gap < 40 and y_gap < 12 and is_valid_name(next_item['text']) and abs(
                        next_item['x'] - name_col_x) < name_col_tolerance + 80:
                    best_name = f"{best_name} {next_item['text']}"

            return best_name

    for item in sorted(row_items, key=lambda x: x['x']):
        text = item['text'].strip()
        if re.match(r'^[0-9]{1,2}\.?', text):
            continue
        if is_valid_name(text):
            if not extract_fees(text):
                return text

    return ''


def assign_to_column_printed(item, column_positions, tolerance=90):
    """Code2 column assignment"""
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


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def detect_column_positions(header_items, all_items):
    """Detect column X positions from header and data"""
    columns = {
        'name': None,
        'company': None,
        'invited': None,
        'fees': None,
        'payment': None,
    }

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
            if len(col_positions) >= 5:
                columns['name'] = col_positions[0]
                columns['company'] = col_positions[1]
                columns['invited'] = col_positions[2]
                columns['fees'] = col_positions[3]
                columns['payment'] = col_positions[4]

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


def is_valid_name(text):
    """Enhanced name validation"""
    text = text.strip()
    if len(text) < 2:
        return False

    if re.match(r'^[\d\s\.\,\-\/\:\(\)\[\]]+', text):
        return False

    reject_patterns = [r'^\d+[-/]\d+[=\-_]+^\d +\s * / -']
    for pattern in reject_patterns:
        if re.search(pattern, text):
            return False

    headers = [
        'name', 'company', 'invited', 'fees', 'payment', 'date', 'note',
        'signature', 'mode', 'visitor', 'sheet', 'brilliance', 'bni',
        'total', 'amount', 'cash', 'online', 'upi', 'card', 'cheque',
        'serial', 'no.', 'sr.', 'visitors'
    ]
    text_lower = text.lower()
    if text_lower in headers:
        return False

    company_keywords = [
        'ltd', 'inc', 'corp', 'pvt', 'llc', 'llp', 'industries',
        'technologies', 'jewellery', 'infotech', 'services'
    ]
    if any(keyword in text_lower for keyword in company_keywords):
        return False

    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < 2:
        return False

    if len(text) > 0:
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < 0.4:
            return False

    if text[0].isdigit() and alpha_count < 3:
        return False

    if '=' in text and re.search(r'[A-Za-z]+=\d', text):
        return False

    return True


def is_company_name(text):
    """Detect company names"""
    text = text.strip()
    if len(text) < 2:
        return False

    company_keywords = [
        'ltd', 'inc', 'corp', 'pvt', 'llc', 'llp', 'co.', 'industries',
        'technologies', 'group', 'jewellery', 'jewel', 'wedding', 'infotech',
        'services', 'land', 'intucent', 'wedingrings', 'fewelly', 'ictoa',
        'tours', 'travels', 'chisel', 'equipments', 'uniclean', 'design',
        'builder', 'management', 'suprabhat', 'engineering', 'idbi', 'navkar',
        'digitqi', 'interior'
    ]
    lower = text.lower()
    return any(keyword in lower for keyword in company_keywords)


def extract_fees(text):
    """Extract fees from text"""
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
    patterns = [r'(\d+)\s*/-', r'=\s*(\d+)^(\d{3,})(\d{3, })']

    for pattern in patterns:
        match = re.search(pattern, clean)
    if match:
        try:
            amount = int(match.group(1))
            if 100 <= amount < 100000:
                return str(amount)
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


def is_signature_or_check(text):
    """Identify signature or checkmark symbols"""
    text = text.strip()
    if text in ['✓', '√', '×', '✗', 'V', 'v', 'x', 'X', '/', '\\', 'w', 'W']:
        return True
    if len(text) <= 2 and not text.isalnum():
        return True
    return False


def merge_similar_text(items, similarity_threshold=0.70, x_tolerance=30):
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
            y_close = abs(item1['y'] - item2['y']) < 10

            if x_close and y_close:
                similarity = SequenceMatcher(None, item1['text'].lower(),
                                             item2['text'].lower()).ratio()
                if similarity > similarity_threshold:
                    similar_group.append(item2)
                    used.add(j)

        best_item = max(similar_group, key=lambda x: (x['score'], len(x['text'])))
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


# ============================================================================
# MAIN UNIFIED FUNCTION - IF-ELSE LOGIC
# ============================================================================

def extract_data_from_image_v2(image_path):
    """
    Main function that detects document type and routes to appropriate extraction logic
    """
    # Detect document type
    doc_type = detect_document_type(image_path)

    # Use if-else to route to appropriate extraction function
    if doc_type == 'handwritten':
        print(">>> Using CODE1.PY extraction logic for HANDWRITTEN document <<<\n")
        return extract_handwritten(image_path)
    else:
        print(">>> Using CODE2.PY extraction logic for PRINTED document <<<\n")
        return extract_printed(image_path)


def test_extraction(image_path):
    """Test the unified extraction"""
    try:
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
    # Test with both images
    test_images = ["visitor.jpeg", "visitor1.jpeg"]

    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'#' * 80}")
            print(f"# Testing: {image_path}")
            print(f"{'#' * 80}")
            df = test_extraction(image_path)
            if df is not None and len(df) > 0:
                print(f"\n✓ Success! Extracted {len(df)} records from {image_path}")

                # Optional: Save to CSV
                output_csv = image_path.replace('.jpeg', '_extracted.csv')
                df.to_csv(output_csv, index=False)
                print(f"✓ Saved to {output_csv}")
            else:
                print(f"✗ Extraction failed or no records found in {image_path}")
        else:
            print(f"✗ Image file not found: {image_path}")