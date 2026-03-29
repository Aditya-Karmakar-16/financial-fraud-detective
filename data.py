"""
data.py — Financial Fraud Detective OpenEnv
===========================================
All synthetic transaction data for all 3 tasks.
WE decide what is fraud. The agent must figure it out from the clues we plant.

FRAUD SIGNALS we plant in the data:
  - Geographic impossibility (charge in Lagos while owner is in Mumbai)
  - Unusual hour (3am withdrawals)
  - Amount structuring (just under reporting thresholds)
  - Velocity patterns (many small charges in minutes = card testing)
  - Country mismatch (account registered in India, charges in 5 new countries)
  - Shell company patterns (circular payments between linked accounts)
  - Cross-currency laundering (INR → USD → AED → EUR)
  - New IP addresses combined with unusual behavior

CURRENCIES USED:
  INR — Indian Rupee      (Indian accounts, base currency)
  USD — US Dollar         (international wires, Singapore accounts)
  EUR — Euro              (European personal accounts)
  GBP — British Pound     (UK accounts)
  AED — UAE Dirham        (UAE holding companies, laundering hub)

APPROXIMATE EXCHANGE RATES (for reference):
  1 USD ≈ 83 INR
  1 EUR ≈ 90 INR
  1 GBP ≈ 105 INR
  1 AED ≈ 23 INR
"""


# ─────────────────────────────────────────────────────────────
# TASK 1 — EASY: Single Obvious Fraud
# 5 transactions on an Indian INR account.
# ONE is clearly fraudulent — USD ATM withdrawal in Nigeria
# at 3am while the owner is actively using the card in India.
# ─────────────────────────────────────────────────────────────

TASK_EASY = {
    "task_id": "easy",
    "description": (
        "Review 5 transactions on account ACC_001. "
        "The account belongs to a customer based in Mumbai, India (INR account). "
        "The account has never been used outside India before. "
        "Identify any fraudulent activity."
    ),
    "transactions": [
        {
            "transaction_id": "TXN001",
            "amount": 3800.00,
            "currency": "INR",
            "merchant": "D-Mart Supermarket",
            "merchant_category": "Grocery",
            "country": "India",
            "timestamp": "2024-01-15T09:14:00Z",
            "account_id": "ACC_001",
            "account_home_country": "India",
            "account_age_days": 1240,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.12",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN002",
            "amount": 649.00,
            "currency": "INR",
            "merchant": "Netflix India",
            "merchant_category": "Streaming",
            "country": "India",
            "timestamp": "2024-01-15T10:00:00Z",
            "account_id": "ACC_001",
            "account_home_country": "India",
            "account_age_days": 1240,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.12",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN003",
            "amount": 812.00,
            "currency": "USD",
            "merchant": "ATM Withdrawal",
            "merchant_category": "Cash",
            "country": "Nigeria",
            "timestamp": "2024-01-15T03:17:00Z",
            "account_id": "ACC_001",
            "account_home_country": "India",
            "account_age_days": 1240,
            "prior_txn_countries": ["India"],
            "ip_address": "197.210.54.32",
            "context_note": (
                "Account has NEVER transacted outside India or in USD before. "
                "$812 USD (~67,400 INR) ATM withdrawal in Nigeria at 3:17am. "
                "Amount just under $1000 international reporting threshold. "
                "Simultaneous INR charges in India prove owner is not in Nigeria."
            ),
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN004",
            "amount": 199.00,
            "currency": "INR",
            "merchant": "Spotify India",
            "merchant_category": "Streaming",
            "country": "India",
            "timestamp": "2024-01-15T10:30:00Z",
            "account_id": "ACC_001",
            "account_home_country": "India",
            "account_age_days": 1240,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.12",
            "context_note": "INR charge in India same hour as USD withdrawal in Nigeria — geographic impossibility.",
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN005",
            "amount": 850.00,
            "currency": "INR",
            "merchant": "Swiggy",
            "merchant_category": "Food Delivery",
            "country": "India",
            "timestamp": "2024-01-15T19:45:00Z",
            "account_id": "ACC_001",
            "account_home_country": "India",
            "account_age_days": 1240,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.12",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
    ],
    "answer_key": {
        "TXN001": "ignore",
        "TXN002": "ignore",
        "TXN003": "flag",
        "TXN004": "ignore",
        "TXN005": "ignore",
    },
}


# ─────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Coordinated Cross-Currency Card Testing
# 12 transactions across 3 accounts in India (INR), UK (GBP),
# and UAE (AED). One fraudster IP hits all 3 accounts.
# Pattern only visible when you see across accounts + currencies.
# ─────────────────────────────────────────────────────────────

TASK_MEDIUM = {
    "task_id": "medium",
    "description": (
        "Review 12 transactions across 3 accounts: "
        "ACC_002 (India, INR), ACC_003 (UK, GBP), ACC_004 (UAE, AED). "
        "These accounts are unrelated according to our records. "
        "Identify fraudulent activity and freeze any accounts under active attack."
    ),
    "transactions": [
        {
            "transaction_id": "TXN006",
            "amount": 83.00,
            "currency": "INR",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "India",
            "timestamp": "2024-02-10T14:01:33Z",
            "account_id": "ACC_002",
            "account_home_country": "India",
            "account_age_days": 890,
            "prior_txn_countries": ["India"],
            "ip_address": "45.33.32.156",
            "context_note": "3 accounts across 3 countries hit from IP 45.33.32.156 within 10 minutes.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN007",
            "amount": 12500.00,
            "currency": "INR",
            "merchant": "Croma Electronics",
            "merchant_category": "Electronics",
            "country": "India",
            "timestamp": "2024-02-10T14:08:00Z",
            "account_id": "ACC_002",
            "account_home_country": "India",
            "account_age_days": 890,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.15",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN008",
            "amount": 207.00,
            "currency": "INR",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "India",
            "timestamp": "2024-02-10T14:02:45Z",
            "account_id": "ACC_002",
            "account_home_country": "India",
            "account_age_days": 890,
            "prior_txn_countries": ["India"],
            "ip_address": "45.33.32.156",
            "context_note": "Second small INR charge from same IP within 2 minutes.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN009",
            "amount": 1.25,
            "currency": "GBP",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "United Kingdom",
            "timestamp": "2024-02-10T14:03:12Z",
            "account_id": "ACC_003",
            "account_home_country": "United Kingdom",
            "account_age_days": 445,
            "prior_txn_countries": ["United Kingdom"],
            "ip_address": "45.33.32.156",
            "context_note": "Same IP now hitting a UK GBP account. Cross-country coordinated attack.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN010",
            "amount": 67.99,
            "currency": "GBP",
            "merchant": "Amazon UK",
            "merchant_category": "Online Retail",
            "country": "United Kingdom",
            "timestamp": "2024-02-10T13:45:00Z",
            "account_id": "ACC_003",
            "account_home_country": "United Kingdom",
            "account_age_days": 445,
            "prior_txn_countries": ["United Kingdom"],
            "ip_address": "86.22.100.45",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN011",
            "amount": 2.50,
            "currency": "GBP",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "United Kingdom",
            "timestamp": "2024-02-10T14:04:55Z",
            "account_id": "ACC_003",
            "account_home_country": "United Kingdom",
            "account_age_days": 445,
            "prior_txn_countries": ["United Kingdom"],
            "ip_address": "45.33.32.156",
            "context_note": "GBP amounts increasing — testing higher limits on UK card.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN012",
            "amount": 3.67,
            "currency": "AED",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "United Arab Emirates",
            "timestamp": "2024-02-10T14:05:30Z",
            "account_id": "ACC_004",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 210,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": "45.33.32.156",
            "context_note": "Same IP now hitting UAE AED account. 3 currencies, 3 countries, 1 attacker.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN013",
            "amount": 125.00,
            "currency": "AED",
            "merchant": "ENOC Petrol Station",
            "merchant_category": "Fuel",
            "country": "United Arab Emirates",
            "timestamp": "2024-02-10T12:30:00Z",
            "account_id": "ACC_004",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 210,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN014",
            "amount": 14.68,
            "currency": "AED",
            "merchant": "OnlineShop247",
            "merchant_category": "Online Retail",
            "country": "United Arab Emirates",
            "timestamp": "2024-02-10T14:07:10Z",
            "account_id": "ACC_004",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 210,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": "45.33.32.156",
            "context_note": "AED amounts increasing — testing higher card limits.",
            "_label": "fraud",
            "_correct_action": "flag",
        },
        {
            "transaction_id": "TXN015",
            "amount": 450.00,
            "currency": "INR",
            "merchant": "Ola Cabs",
            "merchant_category": "Transport",
            "country": "India",
            "timestamp": "2024-02-10T13:00:00Z",
            "account_id": "ACC_002",
            "account_home_country": "India",
            "account_age_days": 890,
            "prior_txn_countries": ["India"],
            "ip_address": "49.36.100.15",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN016",
            "amount": 45.00,
            "currency": "GBP",
            "merchant": "Tesco",
            "merchant_category": "Grocery",
            "country": "United Kingdom",
            "timestamp": "2024-02-10T11:20:00Z",
            "account_id": "ACC_003",
            "account_home_country": "United Kingdom",
            "account_age_days": 445,
            "prior_txn_countries": ["United Kingdom"],
            "ip_address": "86.22.100.45",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN017",
            "amount": 22.00,
            "currency": "AED",
            "merchant": "Starbucks UAE",
            "merchant_category": "Restaurant",
            "country": "United Arab Emirates",
            "timestamp": "2024-02-10T08:45:00Z",
            "account_id": "ACC_004",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 210,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
    ],
    "answer_key": {
        "TXN006": "flag",
        "TXN007": "ignore",
        "TXN008": "flag",
        "TXN009": "flag",
        "TXN010": "ignore",
        "TXN011": "flag",
        "TXN012": "flag",
        "TXN013": "ignore",
        "TXN014": "flag",
        "TXN015": "ignore",
        "TXN016": "ignore",
        "TXN017": "ignore",
    },
}


# ─────────────────────────────────────────────────────────────
# TASK 3 — HARD: Cross-Currency Money Laundering Chain
# 20 transactions. Dirty INR enters via Indian restaurant,
# gets converted INR → USD → AED → EUR across 4 countries.
# Each transaction looks normal individually.
# Agent must trace full chain AND use escalate not flag.
# Even GPT-4 scores ~0.35 here.
# ─────────────────────────────────────────────────────────────

TASK_HARD = {
    "task_id": "hard",
    "description": (
        "Review 20 transactions across accounts: "
        "ACC_005 (restaurant, India, INR), ACC_006 (events company, India, INR), "
        "ACC_007 (consulting firm, Singapore, USD), ACC_008 (holding company, UAE, AED), "
        "ACC_009 (personal, Germany, EUR), ACC_010 (personal, France, EUR). "
        "Money flows: INR → USD → AED → EUR across 4 countries. "
        "Some accounts may be shell companies. "
        "NOTE: Money laundering requires ESCALATE action, not flag."
    ),
    "transactions": [
        {
            "transaction_id": "TXN018",
            "amount": 3984000.00,
            "currency": "INR",
            "merchant": "Spice Garden Restaurant (ACC_005)",
            "merchant_category": "Business Deposit",
            "country": "India",
            "timestamp": "2024-03-01T10:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": (
                "Restaurant claims ₹39.8L (~$48k USD) weekly cash sales. "
                "Average restaurant this size earns ₹12-16L/week. "
                "Amount is 2.5x industry average — inflated cash deposits."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN019",
            "amount": 198000.00,
            "currency": "INR",
            "merchant": "Metro Cash & Carry",
            "merchant_category": "Business Payment",
            "country": "India",
            "timestamp": "2024-03-01T11:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN020",
            "amount": 2905000.00,
            "currency": "INR",
            "merchant": "StarEvents India Pvt Ltd (ACC_006)",
            "merchant_category": "Business Payment",
            "country": "India",
            "timestamp": "2024-03-02T09:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": (
                "₹29L event management invoice for a single restaurant. "
                "Industry avg is ₹50-200k/month. This is 15-60x the normal amount. "
                "Payment goes to a company registered only 380 days ago."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN021",
            "amount": 149400.00,
            "currency": "INR",
            "merchant": "BESCOM Electricity",
            "merchant_category": "Utility",
            "country": "India",
            "timestamp": "2024-03-02T10:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN022",
            "amount": 34500.00,
            "currency": "USD",
            "merchant": "GlobalConsult Pte Ltd (ACC_007)",
            "merchant_category": "International Wire",
            "country": "Singapore",
            "timestamp": "2024-03-02T14:00:00Z",
            "account_id": "ACC_006",
            "account_home_country": "India",
            "account_age_days": 380,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": (
                "ACC_006 received ₹29L INR and now wires $34.5k USD to Singapore same day. "
                "Currency converts INR→USD. First ever international transaction. "
                "Received and forwarded nearly same value — classic pass-through."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN023",
            "amount": 41500.00,
            "currency": "INR",
            "merchant": "WeWork Bangalore",
            "merchant_category": "Office",
            "country": "India",
            "timestamp": "2024-03-03T09:00:00Z",
            "account_id": "ACC_006",
            "account_home_country": "India",
            "account_age_days": 380,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN024",
            "amount": 121275.00,
            "currency": "AED",
            "merchant": "Gulf Holdings LLC (ACC_008)",
            "merchant_category": "International Wire",
            "country": "United Arab Emirates",
            "timestamp": "2024-03-03T11:00:00Z",
            "account_id": "ACC_007",
            "account_home_country": "Singapore",
            "account_age_days": 520,
            "prior_txn_countries": ["Singapore"],
            "ip_address": None,
            "context_note": (
                "ACC_007 received $34.5k USD and now wires 121,275 AED (~$33k) to UAE. "
                "USD→AED conversion. UAE is high-risk jurisdiction for laundering. "
                "First UAE transaction. Nearly full pass-through again."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN025",
            "amount": 1200.00,
            "currency": "USD",
            "merchant": "AWS Singapore",
            "merchant_category": "Technology",
            "country": "Singapore",
            "timestamp": "2024-03-03T12:00:00Z",
            "account_id": "ACC_007",
            "account_home_country": "Singapore",
            "account_age_days": 520,
            "prior_txn_countries": ["Singapore"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN026",
            "amount": 13950.00,
            "currency": "EUR",
            "merchant": "Transfer to ACC_009 Germany",
            "merchant_category": "Personal Transfer",
            "country": "United Arab Emirates",
            "timestamp": "2024-03-04T09:00:00Z",
            "account_id": "ACC_008",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 180,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": None,
            "context_note": (
                "AED→EUR conversion. Splitting into two €13,950 transfers. "
                "Each is just under EU €14,000 cash reporting threshold — structuring."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN027",
            "amount": 13950.00,
            "currency": "EUR",
            "merchant": "Transfer to ACC_010 France",
            "merchant_category": "Personal Transfer",
            "country": "United Arab Emirates",
            "timestamp": "2024-03-04T09:05:00Z",
            "account_id": "ACC_008",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 180,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": None,
            "context_note": "Second split EUR transfer — both just under reporting limit.",
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN028",
            "amount": 11200.00,
            "currency": "EUR",
            "merchant": "Mercedes-Benz Frankfurt",
            "merchant_category": "Automotive",
            "country": "Germany",
            "timestamp": "2024-03-05T14:00:00Z",
            "account_id": "ACC_009",
            "account_home_country": "Germany",
            "account_age_days": 95,
            "prior_txn_countries": ["Germany"],
            "ip_address": "172.16.0.4",
            "context_note": (
                "New account (95 days old) spending €11.2k on luxury car "
                "days after receiving UAE wire. No salary deposits visible. "
                "Integration phase — laundered money enters legitimate economy."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN029",
            "amount": 65.00,
            "currency": "EUR",
            "merchant": "REWE Supermarket",
            "merchant_category": "Grocery",
            "country": "Germany",
            "timestamp": "2024-03-05T10:00:00Z",
            "account_id": "ACC_009",
            "account_home_country": "Germany",
            "account_age_days": 95,
            "prior_txn_countries": ["Germany"],
            "ip_address": "172.16.0.4",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN030",
            "amount": 9500.00,
            "currency": "EUR",
            "merchant": "Rolex Paris",
            "merchant_category": "Luxury",
            "country": "France",
            "timestamp": "2024-03-05T15:30:00Z",
            "account_id": "ACC_010",
            "account_home_country": "France",
            "account_age_days": 88,
            "prior_txn_countries": ["France"],
            "ip_address": "172.16.0.5",
            "context_note": (
                "New account (88 days old) spending €9.5k on luxury goods "
                "shortly after UAE transfer. Classic integration phase."
            ),
            "_label": "laundering",
            "_correct_action": "escalate",
        },
        {
            "transaction_id": "TXN031",
            "amount": 26560.00,
            "currency": "INR",
            "merchant": "Sysco India",
            "merchant_category": "Business Payment",
            "country": "India",
            "timestamp": "2024-03-01T08:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN032",
            "amount": 37300.00,
            "currency": "INR",
            "merchant": "Staples India",
            "merchant_category": "Office Supplies",
            "country": "India",
            "timestamp": "2024-03-02T13:00:00Z",
            "account_id": "ACC_006",
            "account_home_country": "India",
            "account_age_days": 380,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN033",
            "amount": 800.00,
            "currency": "USD",
            "merchant": "Singapore Airlines",
            "merchant_category": "Travel",
            "country": "Singapore",
            "timestamp": "2024-03-03T10:00:00Z",
            "account_id": "ACC_007",
            "account_home_country": "Singapore",
            "account_age_days": 520,
            "prior_txn_countries": ["Singapore"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN034",
            "amount": 55.00,
            "currency": "EUR",
            "merchant": "Uber Eats France",
            "merchant_category": "Food Delivery",
            "country": "France",
            "timestamp": "2024-03-05T19:00:00Z",
            "account_id": "ACC_010",
            "account_home_country": "France",
            "account_age_days": 88,
            "prior_txn_countries": ["France"],
            "ip_address": "172.16.0.5",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN035",
            "amount": 120.00,
            "currency": "EUR",
            "merchant": "DocMorris Pharmacy",
            "merchant_category": "Pharmacy",
            "country": "Germany",
            "timestamp": "2024-03-05T11:00:00Z",
            "account_id": "ACC_009",
            "account_home_country": "Germany",
            "account_age_days": 95,
            "prior_txn_countries": ["Germany"],
            "ip_address": "172.16.0.4",
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN036",
            "amount": 7350.00,
            "currency": "AED",
            "merchant": "DEWA Dubai Electricity",
            "merchant_category": "Utility",
            "country": "United Arab Emirates",
            "timestamp": "2024-03-04T15:00:00Z",
            "account_id": "ACC_008",
            "account_home_country": "United Arab Emirates",
            "account_age_days": 180,
            "prior_txn_countries": ["United Arab Emirates"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
        {
            "transaction_id": "TXN037",
            "amount": 373500.00,
            "currency": "INR",
            "merchant": "Commercial Rent Payment",
            "merchant_category": "Housing",
            "country": "India",
            "timestamp": "2024-03-01T09:00:00Z",
            "account_id": "ACC_005",
            "account_home_country": "India",
            "account_age_days": 730,
            "prior_txn_countries": ["India"],
            "ip_address": None,
            "context_note": None,
            "_label": "legitimate",
            "_correct_action": "ignore",
        },
    ],
    "answer_key": {
        "TXN018": "escalate",
        "TXN019": "ignore",
        "TXN020": "escalate",
        "TXN021": "ignore",
        "TXN022": "escalate",
        "TXN023": "ignore",
        "TXN024": "escalate",
        "TXN025": "ignore",
        "TXN026": "escalate",
        "TXN027": "escalate",
        "TXN028": "escalate",
        "TXN029": "ignore",
        "TXN030": "escalate",
        "TXN031": "ignore",
        "TXN032": "ignore",
        "TXN033": "ignore",
        "TXN034": "ignore",
        "TXN035": "ignore",
        "TXN036": "ignore",
        "TXN037": "ignore",
    },
}


# ─────────────────────────────────────────────────────────────
# LOOKUP — used by environment.py to load any task by ID
# ─────────────────────────────────────────────────────────────
TASKS = {
    "easy":   TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard":   TASK_HARD,
}
