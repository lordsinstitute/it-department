import hashlib
import json
from datetime import datetime

class BlockchainLedger:
    """
    Blockchain-inspired integrity verification system.
    Creates a hash chain for all scan records to ensure evidence integrity.
    """
    
    def __init__(self, ledger_file='integrity_ledger.json'):
        self.ledger_file = ledger_file
        self.ledger = self._load_ledger()
    
    def _load_ledger(self):
        """Load existing ledger or create new one"""
        try:
            with open(self.ledger_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'version': '1.0',
                'created_at': datetime.utcnow().isoformat(),
                'blocks': [],
                'genesis_hash': self._calculate_genesis_hash()
            }
    
    def _save_ledger(self):
        """Save ledger to file"""
        try:
            with open(self.ledger_file, 'w') as f:
                json.dump(self.ledger, f, indent=2)
        except Exception as e:
            print(f"Error saving ledger: {str(e)}")
    
    def _calculate_genesis_hash(self):
        """Create genesis block hash"""
        genesis_data = {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'data': 'Genesis Block'
        }
        return self._hash_data(genesis_data)
    
    def _hash_data(self, data):
        """Generate SHA256 hash of data"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def add_record(self, scan_id, analysis_results, risk_level):
        """
        Add a record to the blockchain ledger.
        Returns the hash chain information.
        """
        # Create block data
        block_data = {
            'scan_id': scan_id,
            'timestamp': datetime.utcnow().isoformat(),
            'risk_level': risk_level,
            'threat_count': len(analysis_results.get('threats', [])),
            'file_hash': analysis_results.get('file_hash', ''),
            'threat_summary': {
                'critical': len([t for t in analysis_results.get('threats', []) if t['severity'] == 'Critical']),
                'high': len([t for t in analysis_results.get('threats', []) if t['severity'] == 'High']),
                'medium': len([t for t in analysis_results.get('threats', []) if t['severity'] == 'Medium'])
            }
        }
        
        # Calculate current hash
        current_hash = self._hash_data(block_data)
        
        # Get previous hash
        if self.ledger['blocks']:
            previous_hash = self.ledger['blocks'][-1]['hash']
        else:
            previous_hash = self.ledger['genesis_hash']
        
        # Create block with chain reference
        block = {
            'index': len(self.ledger['blocks']) + 1,
            'data': block_data,
            'hash': current_hash,
            'previous_hash': previous_hash,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add to ledger
        self.ledger['blocks'].append(block)
        self._save_ledger()
        
        # Return hash chain info
        return {
            'current_hash': current_hash,
            'previous_hash': previous_hash,
            'chain_index': block['index'],
            'timestamp': block['timestamp'],
            'verified': self.verify_chain()
        }
    
    def verify_chain(self):
        """Verify integrity of the entire hash chain"""
        if not self.ledger['blocks']:
            return True
        
        # Verify genesis connection
        if self.ledger['blocks'][0]['previous_hash'] != self.ledger['genesis_hash']:
            return False
        
        # Verify each block
        for i, block in enumerate(self.ledger['blocks']):
            # Verify hash
            calculated_hash = self._hash_data(block['data'])
            if calculated_hash != block['hash']:
                return False
            
            # Verify previous hash reference
            if i > 0:
                if block['previous_hash'] != self.ledger['blocks'][i-1]['hash']:
                    return False
        
        return True
    
    def get_chain_proof(self, scan_id):
        """Get the proof of record for a specific scan"""
        for block in self.ledger['blocks']:
            if block['data']['scan_id'] == scan_id:
                return {
                    'scan_id': scan_id,
                    'hash': block['hash'],
                    'previous_hash': block['previous_hash'],
                    'chain_index': block['index'],
                    'timestamp': block['timestamp'],
                    'chain_verified': self.verify_chain()
                }
        return None
