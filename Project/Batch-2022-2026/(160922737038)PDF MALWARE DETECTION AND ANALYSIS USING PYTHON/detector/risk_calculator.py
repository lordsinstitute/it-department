class RiskCalculator:
    """Calculate risk level based on analysis findings"""
    
    THREAT_WEIGHTS = {
        'Critical': 4,
        'High': 3,
        'Medium': 2,
        'Low': 1,
        'Info': 0,
        'Warning': 0.5
    }
    
    THREAT_MULTIPLIERS = {
        'JavaScript Detected': 1.5,
        'Auto-execute Action': 1.8,
        'Embedded Executable': 2.0,
        'Known Malware Signature': 2.5,
        'Dangerous Command Found': 2.0,
        'Additional Actions Detected': 1.6,
        'Rich Media/Flash Detected': 1.7,
        'Suspicious URL': 1.3,
    }
    
    def calculate_risk(self, analysis_results):
        """
        Calculate overall risk level: Low, Medium, High, or Critical
        """
        threats = analysis_results.get('threats', [])
        
        if not threats:
            return 'Low'
        
        # Calculate threat score
        threat_score = 0
        critical_count = 0
        high_count = 0
        
        for threat in threats:
            severity = threat.get('severity', 'Info')
            threat_type = threat.get('type', '')
            
            weight = self.THREAT_WEIGHTS.get(severity, 0)
            multiplier = self.THREAT_MULTIPLIERS.get(threat_type, 1.0)
            threat_score += weight * multiplier
            
            if severity == 'Critical':
                critical_count += 1
            elif severity == 'High':
                high_count += 1
        
        # Determine risk level
        if critical_count >= 1 or threat_score >= 8:
            return 'Critical'
        elif high_count >= 2 or threat_score >= 5:
            return 'High'
        elif threat_score >= 2.5:
            return 'Medium'
        else:
            return 'Low'
    
    def get_risk_color(self, risk_level):
        """Get color for risk level"""
        colors = {
            'Low': '#28a745',
            'Medium': '#ffc107',
            'High': '#fd7e14',
            'Critical': '#dc3545'
        }
        return colors.get(risk_level, '#6c757d')
    
    def get_risk_icon(self, risk_level):
        """Get icon for risk level"""
        icons = {
            'Low': 'check-circle',
            'Medium': 'exclamation-circle',
            'High': 'warning',
            'Critical': 'times-circle'
        }
        return icons.get(risk_level, 'question-circle')
