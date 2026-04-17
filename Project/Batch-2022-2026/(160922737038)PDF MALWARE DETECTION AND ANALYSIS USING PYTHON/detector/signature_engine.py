import json

class SignatureEngine:
    """Malware signature database and matching engine"""
    
    SIGNATURES = {
        'pdf_exploits': {
            'CVE-2010-0188': {
                'name': 'Adobe Reader Launch Exploit',
                'pattern': 'Launch\s*<<',
                'severity': 'Critical'
            },
            'CVE-2008-2992': {
                'name': 'Adobe Reader Collab Exploit',
                'pattern': 'collab\.collectEmailInfo',
                'severity': 'Critical'
            },
            'CVE-2009-0927': {
                'name': 'Adobe Reader getIcon Exploit',
                'pattern': 'getIcon\s*\(',
                'severity': 'Critical'
            }
        },
        'javascript_exploits': {
            'shellcode_pattern': {
                'name': 'Potential Shellcode',
                'pattern': r'\\x[0-9a-f]{2}',
                'severity': 'High'
            },
            'obfuscation': {
                'name': 'Obfuscated JavaScript',
                'pattern': 'String\.fromCharCode',
                'severity': 'High'
            },
            'heap_spray': {
                'name': 'Potential Heap Spray Attack',
                'pattern': 'new Array|while\s*\(',
                'severity': 'High'
            }
        },
        'behavioral_indicators': {
            'network_access': {
                'name': 'Network Access Attempt',
                'pattern': 'http|ftp|smb',
                'severity': 'Medium'
            },
            'file_operations': {
                'name': 'File Operations',
                'pattern': 'CreateObject|FSO|WriteFile',
                'severity': 'High'
            },
            'registry_access': {
                'name': 'Registry Access',
                'pattern': 'WScript\.Shell|RegRead|RegWrite',
                'severity': 'High'
            }
        }
    }
    
    @classmethod
    def get_signatures(cls):
        """Return all signatures"""
        return cls.SIGNATURES
    
    @classmethod
    def get_signature_names(cls):
        """Get all signature names"""
        names = []
        for category in cls.SIGNATURES.values():
            names.extend(category.keys())
        return names
