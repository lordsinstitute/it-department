from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from datetime import datetime
import json
from io import BytesIO

class ReportGenerator:
    """Generate PDF reports from scan results"""
    
    def generate_report(self, scan, analysis_result, analysis_data):
        """Generate a comprehensive PDF report"""
        buffer = BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        title = Paragraph('PDF Malware Scanner Report', title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        summary_heading = Paragraph('Executive Summary', heading_style)
        elements.append(summary_heading)
        
        risk_color = self._get_risk_color(scan.risk_level)
        summary_data = [
            ['Scan Date', scan.created_at.strftime('%Y-%m-%d %H:%M:%S')],
            ['Filename', scan.filename],
            ['Risk Level', scan.risk_level],
            ['Threats Detected', str(scan.threat_count)],
            ['File Size', f"{analysis_data.get('filesize', 0) / 1024:.2f} KB"],
            ['File Hash (SHA256)', analysis_data.get('file_hash', 'N/A')[:32] + '...'],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Threats Details
        threats_heading = Paragraph('Detected Threats', heading_style)
        elements.append(threats_heading)
        
        threats = analysis_data.get('threats', [])
        if threats:
            threats_data = [['Type', 'Description', 'Severity']]
            for threat in threats:
                threats_data.append([
                    threat.get('type', 'Unknown'),
                    threat.get('description', 'N/A')[:50],
                    threat.get('severity', 'Unknown')
                ])
            
            threats_table = Table(threats_data, colWidths=[1.5*inch, 3*inch, 1.5*inch])
            threats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(threats_table)
        else:
            no_threats = Paragraph('No threats detected.', styles['Normal'])
            elements.append(no_threats)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Metadata
        metadata_heading = Paragraph('File Metadata', heading_style)
        elements.append(metadata_heading)
        
        metadata = analysis_data.get('metadata', {})
        metadata_data = [['Property', 'Value']]
        for key, value in metadata.items():
            metadata_data.append([key.replace('_', ' ').title(), str(value)])
        
        if len(metadata_data) > 1:
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(metadata_table)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Integrity Hash
        hash_heading = Paragraph('Evidence Integrity Verification', heading_style)
        elements.append(hash_heading)
        
        hash_chain = json.loads(analysis_result.hash_chain) if analysis_result else {}
        hash_info = [
            ['Hash Value', analysis_result.hash_value if analysis_result else 'N/A'],
            ['Chain Index', str(hash_chain.get('chain_index', 'N/A'))],
            ['Chain Verified', 'Yes' if hash_chain.get('verified') else 'No'],
            ['Timestamp', hash_chain.get('timestamp', 'N/A')]
        ]
        
        hash_table = Table(hash_info, colWidths=[2*inch, 4*inch])
        hash_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(hash_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Footer
        footer_text = Paragraph(
            f'<i>Report generated by PDF Malware Scanner on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</i>',
            styles['Normal']
        )
        elements.append(footer_text)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level"""
        colors_map = {
            'Low': colors.HexColor('#28a745'),
            'Medium': colors.HexColor('#ffc107'),
            'High': colors.HexColor('#fd7e14'),
            'Critical': colors.HexColor('#dc3545')
        }
        return colors_map.get(risk_level, colors.HexColor('#6c757d'))
