from src.agent.policy import decide

def test_low_conf_escalates():
    assert decide('refund',0.4,0.8)[0]=='ESCALATE'

def test_sensitive_escalates():
    assert decide('account_issue',0.9,0.8)[0]=='ESCALATE'

def test_good_low_risk_handles():
    assert decide('refund',0.9,0.8)[0]=='AUTO-HANDLE'
