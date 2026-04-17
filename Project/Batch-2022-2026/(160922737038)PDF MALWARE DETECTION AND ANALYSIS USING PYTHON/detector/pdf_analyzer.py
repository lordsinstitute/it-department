import os
import json
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PDFAnalyzer:
    """
    Core PDF analysis engine with signature-based detection.
    Analyzes PDFs for malware indicators including:
    - Suspicious JavaScript
    - Embedded executables
    - Malicious metadata
    - Exploit patterns
    """
    
    def __init__(self):
        self.signatures = self._load_signatures()
    
    def _load_signatures(self):
        """Load malware signature database"""
        return {
            'javascript_patterns': [
                r'eval\s*\(',
                r'unescape\s*\(',
                r'replace\s*\(',
                r'openAction',
                r'AA\s*<<',
                r'JavaScript\s*',
                r'RichMedia',
                r'Flash',
                r'XObject\s*<<.*?ObjStm'
            ],
            'dangerous_commands': [
                'powershell',
                'cmd.exe',
                'rundll32',
                'regsvr32',
                'cscript',
                'wscript',
                'certutil',
                'bitsadmin',
                'mshta'
            ],
            'suspicious_urls': [
                r'http[s]?://[^\s]+\.exe',
                r'http[s]?://[^\s]+\.(bat|cmd|scr|vbs|js)',
                r'ftp://[^\s]+',
                r'smb://[^\s]+'
            ],
            'malicious_signatures': [
                ('Exploit.PDF.CVE-2010-0188', b'\x25PDF.*?Launch\s*<<'),
                ('Exploit.PDF.Embedded.Exe', b'%PDF.*?EmbeddedFile.*?\.exe'),
                ('Exploit.PDF.OpenAction', b'%PDF.*?OpenAction\s*<<'),
            ]
        }
    
    def analyze_file(self, filepath):
        """
        Main analysis method
        """
        try:
            analysis_results = {
                'filename': os.path.basename(filepath),
                'filesize': os.path.getsize(filepath),
                'analysis_date': datetime.utcnow().isoformat(),
                'threats': [],
                'metadata': {},
                'file_hash': self._calculate_hash(filepath)
            }
            
            # Read file
            try:
                with open(filepath, 'rb') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {filepath}: {str(e)}")
                analysis_results['threats'].append({
                    'type': 'File Read Error',
                    'description': f'Failed to read file: {str(e)}',
                    'severity': 'Warning'
                })
                return analysis_results
            
            # Validate PDF
            if not self._is_valid_pdf(file_content):
                analysis_results['threats'].append({
                    'type': 'Invalid PDF',
                    'description': 'File does not have valid PDF signature',
                    'severity': 'Warning'
                })
            
            # Extract and analyze metadata
            metadata = self._extract_metadata(file_content)
            analysis_results['metadata'] = metadata
            
            # Scan for JavaScript
            js_threats = self._scan_javascript(file_content)
            analysis_results['threats'].extend(js_threats)
            
            # Scan for dangerous patterns
            pattern_threats = self._scan_patterns(file_content)
            analysis_results['threats'].extend(pattern_threats)
            
            # Scan for embedded executables
            exe_threats = self._scan_embedded_files(file_content)
            analysis_results['threats'].extend(exe_threats)
            
            # Scan for malicious signatures
            sig_threats = self._scan_signatures(file_content)
            analysis_results['threats'].extend(sig_threats)
            
            # Scan for suspicious URLs
            url_threats = self._scan_urls(file_content)
            analysis_results['threats'].extend(url_threats)
            
            logger.info(f"Analysis complete for {os.path.basename(filepath)}: {len(analysis_results['threats'])} threats found")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis error for {filepath}: {str(e)}")
            return {
                'filename': os.path.basename(filepath),
                'error': str(e),
                'threats': [{
                    'type': 'Analysis Error',
                    'description': f'Analysis failed: {str(e)}',
                    'severity': 'Error'
                }]
            }
    
    def _calculate_hash(self, filepath):
        """Calculate SHA256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation error: {str(e)}")
            return "error"
    
    def _is_valid_pdf(self, content):
        """Check if file is valid PDF"""
        return content[:4] == b'%PDF'
    
    def _extract_metadata(self, content):
        """Extract PDF metadata"""
        metadata = {}
        try:
            content_str = content.decode('latin1', errors='ignore')
            
            # Simple metadata extraction
            if '/Producer' in content_str:
                start = content_str.find('/Producer')
                end = content_str.find('>', start)
                if start != -1 and end != -1:
                    metadata['producer'] = content_str[start:end].strip()
            
            if '/Creator' in content_str:
                start = content_str.find('/Creator')
                end = content_str.find('>', start)
                if start != -1 and end != -1:
                    metadata['creator'] = content_str[start:end].strip()
            
            if '/Author' in content_str:
                start = content_str.find('/Author')
                end = content_str.find('>', start)
                if start != -1 and end != -1:
                    metadata['author'] = content_str[start:end].strip()
            
            # Check for suspicious metadata
            if 'JavaScript' in content_str:
                metadata['has_javascript'] = True
            
            if 'OpenAction' in content_str:
                metadata['has_open_action'] = True
            
            if '/AA' in content_str:
                metadata['has_additional_actions'] = True
        
        except Exception as e:
            logger.error(f"Metadata extraction error: {str(e)}")
        
        return metadata
    
    def _scan_javascript(self, content):
        """Scan for JavaScript content"""
        threats = []
        try:
            content_str = content.decode('latin1', errors='ignore')
            
            if '/JavaScript' in content_str or 'javascript' in content_str.lower():
                threats.append({
                    'type': 'JavaScript Detected',
                    'description': 'PDF contains JavaScript which could execute malicious code',
                    'severity': 'High',
                    'pattern': 'JavaScript'
                })
        except Exception as e:
            logger.error(f"JavaScript scan error: {str(e)}")
        
        return threats
    
    def _scan_patterns(self, content):
        """Scan for dangerous patterns"""
        threats = []
        try:
            content_str = content.decode('latin1', errors='ignore')
            
            for pattern in self.signatures['dangerous_commands']:
                if pattern.lower() in content_str.lower():
                    threats.append({
                        'type': 'Dangerous Command Found',
                        'description': f'Found reference to dangerous command: {pattern}',
                        'severity': 'Critical',
                        'pattern': pattern
                    })
            
            if 'OpenAction' in content_str:
                threats.append({
                    'type': 'Auto-execute Action',
                    'description': 'PDF has OpenAction which could execute on open',
                    'severity': 'High',
                    'pattern': 'OpenAction'
                })
            
            if '/AA' in content_str or 'Additional' in content_str:
                threats.append({
                    'type': 'Additional Actions Detected',
                    'description': 'PDF contains additional actions that could execute malicious code',
                    'severity': 'High',
                    'pattern': '/AA'
                })
            
            if 'RichMedia' in content_str or 'Flash' in content_str:
                threats.append({
                    'type': 'Rich Media/Flash Detected',
                    'description': 'PDF contains embedded RichMedia or Flash which could harbor exploits',
                    'severity': 'High',
                    'pattern': 'RichMedia/Flash'
                })
        
        except Exception as e:
            logger.error(f"Pattern scan error: {str(e)}")
        
        return threats
    
    def _scan_embedded_files(self, content):
        """Scan for embedded executable files"""
        threats = []
        try:
            content_str = content.decode('latin1', errors='ignore')
            
            executable_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js', '.jar', '.zip']
            
            for ext in executable_extensions:
                if ext in content_str.lower():
                    threats.append({
                        'type': 'Embedded Executable',
                        'description': f'PDF may contain embedded file with extension: {ext}',
                        'severity': 'Critical',
                        'pattern': f'Embedded{ext}'
                    })
        
        except Exception as e:
            logger.error(f"Embedded file scan error: {str(e)}")
        
        return threats
    
    def _scan_signatures(self, content):
        """Scan for known malware signatures"""
        threats = []
        try:
            for sig_name, sig_pattern in self.signatures['malicious_signatures']:
                if sig_pattern in content:
                    threats.append({
                        'type': 'Known Malware Signature',
                        'description': f'Matched signature: {sig_name}',
                        'severity': 'Critical',
                        'signature': sig_name
                    })
        
        except Exception as e:
            logger.error(f"Signature scan error: {str(e)}")
        
        return threats
    
    def _scan_urls(self, content):
        """Scan for suspicious URLs"""
        threats = []
        try:
            import re
            content_str = content.decode('latin1', errors='ignore')
            
            for pattern in self.signatures['suspicious_urls']:
                matches = re.findall(pattern, content_str)
                if matches:
                    threats.append({
                        'type': 'Suspicious URL',
                        'description': f'Found suspicious URL pattern: {matches[0] if matches else pattern}',
                        'severity': 'Medium',
                        'pattern': pattern
                    })
        
        except Exception as e:
            logger.error(f"URL scan error: {str(e)}")
        
        return threats
