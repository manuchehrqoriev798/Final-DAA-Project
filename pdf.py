#!/usr/bin/env python3
"""
Generate PDF from result.txt with red benchmarks highlighted in red color
"""

import os
import sys

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not available. Install with: pip install reportlab")


def is_red_benchmark(benchmark_name):
    """Check if a benchmark requires complement graph conversion"""
    red_benchmarks = {
        'gen200_p0.9_44',
        'phat300_1_c',
        'phat300_2_c',
        'mann_a27',
        'sanr400_0.5',
        'sanr400_0.7'
    }
    return benchmark_name in red_benchmarks


def read_result_txt():
    """Read result.txt and parse the table"""
    result_file = os.path.join(os.path.dirname(__file__), 'result.txt')
    
    if not os.path.exists(result_file):
        print(f"Error: {result_file} not found")
        return None
    
    headers = []
    rows = []
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|--'):
                continue
            
            parts = [p.strip() for p in line.split('|')]
            # Remove empty first and last elements
            parts = [p for p in parts if p]
            
            if not parts:
                continue
            
            # Check if this is the header
            if 'Benchmarks' in parts[0] or (len(parts) > 1 and ('V' in parts[1] or '|V|' in line)):
                # Fix |V| column name - when split by |, |V| becomes just V
                fixed_parts = []
                for i, p in enumerate(parts):
                    if i == 1 and p.strip() == 'V':
                        fixed_parts.append('|V|')
                    else:
                        fixed_parts.append(p)
                headers = fixed_parts
            else:
                # Check if this is a data row
                if len(parts) >= 2 and parts[0] and not parts[0].startswith('-'):
                    rows.append(parts)
    
    return headers, rows


def generate_pdf():
    """Generate PDF from result.txt with red benchmarks highlighted"""
    if not HAS_REPORTLAB:
        print("Error: reportlab is required. Install with: pip install reportlab")
        return False
    
    # Read data
    headers, rows = read_result_txt()
    
    if not headers or not rows:
        print("Error: Could not read data from result.txt")
        return False
    
    # Create PDF with minimal margins to fit on one page
    pdf_file = os.path.join(os.path.dirname(__file__), 'results.pdf')
    doc = SimpleDocTemplate(
        pdf_file, 
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.4*inch
    )
    story = []
    
    # Title with R explanation
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=2,
        alignment=1  # Center
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=4,
        alignment=1  # Center
    )
    
    title = Paragraph("Vertex Cover Algorithm - Benchmark Results", title_style)
    subtitle = Paragraph("R = Approximation Ratio (τ(G) / Optimal MVC)", subtitle_style)
    story.append(title)
    story.append(subtitle)
    story.append(Spacer(1, 0.05*inch))
    
    # Prepare table data - use headers as read from file
    # Clean headers: remove empty strings and strip whitespace
    clean_headers = []
    for h in headers:
        h_clean = h.strip()
        if h_clean:
            clean_headers.append(h_clean)
    
    # Check if R column exists in headers
    has_r_column = 'R' in clean_headers
    
    # If R column doesn't exist, find τ(G) and insert R after it
    if not has_r_column:
        tau_index = None
        for idx, h in enumerate(clean_headers):
            if 'τ(G)' in h:
                tau_index = idx
                break
        if tau_index is not None:
            clean_headers.insert(tau_index + 1, "R")
    
    table_data = [clean_headers]
    
    # Process rows and identify red benchmarks, calculate ratios
    red_row_indices = []
    green_cell_indices = []  # Store (row, col) tuples for green cells
    
    for i, row in enumerate(rows):
        # Check if first column contains "(red)" marker
        benchmark_name = row[0] if row else ""
        is_red = False
        
        if benchmark_name.startswith("(red)"):
            is_red = True
            # Remove the (red) marker for display - clean up name
            cleaned_name = benchmark_name.replace("(red)", "").strip()
            row[0] = cleaned_name
        else:
            # Also check by name in case marker is missing
            is_red = is_red_benchmark(benchmark_name)
            # Clean up benchmark name - remove any trailing underscores or extra spaces
            row[0] = benchmark_name.strip()
        
        if is_red:
            red_row_indices.append(i + 1)  # +1 because headers are row 0
        
        # Clean row cells and prepare row data
        clean_row = []
        for cell in row:
            cell_clean = cell.strip() if isinstance(cell, str) else str(cell).strip() if cell else ""
            clean_row.append(cell_clean)
        
        # Find indices for key columns in clean_headers
        r_col_idx = None
        tau_col_idx = None
        opt_col_idx = None
        
        for idx, h in enumerate(clean_headers):
            if h == 'R':
                r_col_idx = idx
            elif 'τ(G)' in h:
                tau_col_idx = idx
            elif 'Optimal' in h or 'MVC' in h:
                opt_col_idx = idx
        
        # Calculate or extract ratio
        ratio_str = ""
        ratio_value = None
        
        # If R column exists in headers, try to get it from row
        if r_col_idx is not None and len(clean_row) > r_col_idx:
            r_val = clean_row[r_col_idx].strip()
            if r_val and r_val != '':
                try:
                    ratio_value = float(r_val)
                    ratio_str = r_val
                except ValueError:
                    pass
        
        # If ratio not found and we have data, calculate it
        if not ratio_str and tau_col_idx is not None and opt_col_idx is not None:
            try:
                if len(clean_row) > tau_col_idx and len(clean_row) > opt_col_idx:
                    tau_g_str = clean_row[tau_col_idx].strip()
                    opt_mvc_str = clean_row[opt_col_idx].strip()
                    
                    if tau_g_str and opt_mvc_str:
                        tau_g = int(tau_g_str)
                        opt_mvc = int(opt_mvc_str)
                        if opt_mvc > 0:
                            ratio_value = tau_g / opt_mvc
                            ratio_str = f"{ratio_value:.3f}"
            except (ValueError, ZeroDivisionError, IndexError):
                pass
        
        # Ensure row matches header structure
        while len(clean_row) < len(clean_headers):
            clean_row.append("")
        
        # If R column exists in headers, make sure it's populated in row
        if r_col_idx is not None:
            if len(clean_row) <= r_col_idx:
                clean_row.append(ratio_str)
            else:
                if not clean_row[r_col_idx] or not clean_row[r_col_idx].strip():
                    clean_row[r_col_idx] = ratio_str
        
        # Check for perfect match (ratio = 1.0) for green highlighting
        if ratio_value is not None and r_col_idx is not None:
            if abs(ratio_value - 1.0) < 0.001:
                green_cell_indices.append((i + 1, r_col_idx))
        
        table_data.append(clean_row)
    
    # Create table with optimized column widths to fit on one page
    # Calculate column widths based on available page width (7.5 inches with margins)
    available_width = 7.5 * inch
    col_widths = [
        1.1 * inch,  # Benchmarks
        0.35 * inch,  # |V|
        0.7 * inch,  # Optimal MVC
        0.45 * inch,  # τ(G)
        0.5 * inch,  # R (Approximation Ratio)
        0.45 * inch,  # α(G)
        0.45 * inch,  # ω(G)
        0.7 * inch,  # t
    ]
    
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Define table style - compact for one page
    table_style = TableStyle([
        # Header style
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#2E5090')),
        
        # General cell style - increased row height by 10% (3 -> 3.3, rounded to 4)
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ])
    
    # Highlight red benchmark rows
    for row_idx in red_row_indices:
        table_style.add('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#FFE6E6'))
        table_style.add('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.HexColor('#CC0000'))
        table_style.add('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold')
    
    # Highlight green cells where ratio == 1.0 (perfect match)
    for row_idx, col_idx in green_cell_indices:
        table_style.add('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#90EE90'))  # Light green
        table_style.add('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), 'Helvetica-Bold')
    
    table.setStyle(table_style)
    story.append(table)
    
    # Build PDF
    doc.build(story)
    
    print(f"\n✓ PDF generated successfully: {pdf_file}")
    return True


if __name__ == '__main__':
    print("="*80)
    print("Generating PDF from result.txt")
    print("="*80)
    
    success = generate_pdf()
    
    if success:
        print("\n" + "="*80)
        print("PDF generation complete!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("PDF generation failed!")
        print("="*80)
        sys.exit(1)

