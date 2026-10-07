"""Seeds for synthetic decision data: task families, domains, input styles and contrastive edits.

Each generated template is one (family, domain, style) draw. The teacher invents the question, options and inputs.
Nothing here comes from DecideBench. The families cover the general kinds of decisions people send to Jev-style
models, which is a much wider set than any one benchmark.
"""

# name, kind (choice | noul | score), what the decision is, hint for the options
FAMILIES = [
    ("agent_tool_routing", "choice", "which tool an AI agent should call next for the user's request",
     "4-6 tools with clear scopes, one of them ask_clarifying_question"),
    ("agent_action_approval", "choice", "how to handle an AI agent's proposed action (shell command, payment, email, "
     "database write, file change, API call) under a risk policy stated in the options",
     "3 options such as auto_approve / require_human_review / block, each with its policy criteria"),
    ("claim_verification", "choice", "whether evidence supports, contradicts or says nothing about a claim; "
     "the input holds both the evidence and the claim", "supported / contradicted / not_enough_info"),
    ("numeric_claim_check", "choice", "whether a claim with numbers, units, percentages or dates matches a source "
     "table or passage in the input", "supported / contradicted / not_enough_info"),
    ("policy_moderation", "choice", "how to moderate a post under a written community policy with exceptions "
     "(quoting abuse to report it, satire, friendly teasing, news reporting)",
     "4-5 actions; the question states the policy in one or two sentences"),
    ("policy_date_window", "choice", "a ruling under a written policy with day-count windows (returns, refunds, "
     "warranty, trial cancellation, late fees, rebooking, leave requests); the input holds the policy and the request "
     "with exact dates", "4-5 rulings such as full refund / partial / deny / escalate"),
    ("policy_amount_threshold", "choice", "a ruling under a written policy with money or quantity limits (expense "
     "approval, credit limit, free shipping, bulk discount, overtime); the input holds the policy and the request",
     "3-5 rulings"),
    ("multi_condition_eligibility", "choice", "eligibility that depends on two or three conditions at once "
     "(age, residency, membership, tenure, plan tier); the input holds the rules and the applicant",
     "eligible / not_eligible / needs_more_info, or tiers"),
    ("review_rating", "score", "the reviewer's overall sentiment, including mixed reviews, sarcasm, negation and "
     "comparisons", "ordered levels from very negative to very positive"),
    ("aspect_sentiment", "choice", "the sentiment toward one named aspect of a product in a review that mentions "
     "several aspects", "positive / negative / mixed / not_mentioned"),
    ("stance_detection", "choice", "the author's stance toward a named proposal or target", "favor / against / neutral"),
    ("support_intent", "choice", "the main intent of a customer message", "5-6 intents including none_of_these"),
    ("bug_severity", "choice", "the severity of a bug or incident report, judged by scope, customer impact and "
     "workarounds", "ordered levels such as not_a_bug / minor / degraded / outage"),
    ("helpdesk_routing", "choice", "which team should own an internal IT or operations ticket", "4-6 teams"),
    ("spam_phishing", "choice", "whether a message is legitimate, spam, phishing or a scam",
     "legitimate / spam / phishing / scam"),
    ("prompt_injection", "noul", "whether text an AI agent read (web page, email, tool output, document) tries to "
     "make the agent ignore its instructions or take an action the user did not ask for", "true / false"),
    ("pii_presence", "noul", "whether a text contains a named category of personal data (for example a full "
     "payment card number, a home address or a government ID number)", "true / false"),
    ("lead_qualification", "score", "how qualified a sales lead is against a stated rubric (budget, authority, need, "
     "timeline)", "ordered levels"),
    ("email_triage", "choice", "what the recipient needs to do with an email", "reply_needed / action_required / "
     "fyi_only / ignore"),
    ("search_relevance", "score", "how relevant a document is to a search query; the input holds both",
     "ordered levels from irrelevant to perfect match"),
    ("duplicate_detection", "noul", "whether two tickets, questions or bug reports in the input describe the same "
     "underlying issue", "true / false"),
    ("entailment", "choice", "whether a premise entails, contradicts or is neutral to a hypothesis; the input holds "
     "both", "entails / contradicts / neutral"),
    ("code_review", "choice", "the review decision for a described code change", "approve / request_changes / "
     "needs_security_review"),
    ("log_alert", "choice", "how to handle a log or monitoring event", "ignore / ticket / page_on_call"),
    ("contract_clause", "choice", "how risky a contract clause is for the customer signing it", "standard / "
     "negotiate / reject"),
    ("hr_policy_request", "choice", "a ruling on an employee request under a stated HR policy (leave, remote work, "
     "equipment, expenses)", "approve / deny / needs_manager / needs_more_info"),
    ("insurance_coverage", "choice", "whether a claim is covered under the policy text given in the input",
     "covered / not_covered / partially_covered / needs_more_info"),
    ("payment_risk", "choice", "how to handle a payment or refund request given risk signals", "approve / review / "
     "decline"),
    ("account_security", "choice", "how to classify an account security event", "benign / suspicious / compromised"),
    ("task_completion", "choice", "whether an AI assistant's final reply fully completed the user's request; the "
     "input holds the request and the reply", "complete / partial / failed / refused"),
    ("answer_groundedness", "choice", "whether an assistant's answer is grounded in the retrieved context given in "
     "the input", "grounded / partially_grounded / ungrounded"),
    ("instruction_following", "noul", "whether a response obeys every explicit constraint in the instructions "
     "(length, format, language, things to avoid); the input holds both", "true / false"),
    ("toxicity", "score", "how toxic a message is", "ordered levels from none to severe"),
    ("urgency", "score", "how urgent a message is for the team receiving it", "ordered levels"),
    ("scheduling_intent", "choice", "what a message asks to do with a meeting or booking", "schedule / reschedule / "
     "cancel / confirm / none"),
    ("order_issue_resolution", "choice", "the next step for a customer order issue, using order dates, tracking "
     "and stock details in the input", "wait / reship / refund / escalate"),
    ("access_request", "choice", "a ruling on a request for access to a system or dataset under a role-based policy "
     "stated in the question or input", "grant / grant_read_only / deny / needs_approval"),
    ("schedule_conflict", "noul", "whether a proposed event conflicts with a calendar listed in the input (time "
     "arithmetic, time zones, durations)", "true / false"),
    ("compliance_check", "choice", "whether a message breaks a stated communication rule (promising returns, "
     "medical claims, sharing confidential data)", "compliant / violation / needs_review"),
    ("content_category", "choice", "which category a piece of content belongs to", "5-6 categories with clear "
     "boundaries"),
    # Added 2026-10-05: long option lists, multi-step reasoning, arithmetic and structured records are part of
    # the decision-model job (Jev documents up to 255 options and lists numbers as a weak spot).
    ("many_option_intent", "choice", "which one of many fine-grained intents a short customer message expresses; "
     "several options are easy to confuse", "10-18 specific intents with snake_case keys"),
    ("many_option_routing", "choice", "which one of many teams, queues or tools should handle a request",
     "10-16 options"),
    ("multi_hop_relations", "choice", "a question that needs two or more facts from the input chained together "
     "(who reports to whom, which team owns the service that failed, which plan applies through the customer's "
     "company)", "3-5 options; one distractor is the answer you get by using only one of the facts"),
    ("multi_hop_check", "noul", "whether a statement follows when two or more facts in the input are combined",
     "true / false"),
    ("arithmetic_check", "noul", "whether totals, sums, differences, percentages or unit conversions stated in the "
     "input are correct", "true / false"),
    ("arithmetic_threshold", "choice", "a decision that needs a small calculation from numbers in the input (items "
     "summed against a limit, a percentage change against a threshold, hours worked against an overtime rule)",
     "3-4 options"),
    ("structured_record_triage", "score", "how severe or urgent a structured record is (a JSON alert, agent trace or "
     "invoice whose fields must be weighed together)", "ordered levels"),
    ("structured_record_decision", "choice", "the action to take on a structured JSON record (agent trace, security "
     "alert, invoice) based on several of its fields", "3-5 actions"),
]
NEW_FAMILIES = [f[0] for f in FAMILIES[-8:]]

DOMAINS = [
    "online retail", "consumer banking", "developer tools SaaS", "mobile telecom", "airline travel", "hotels",
    "food delivery", "clinic front-desk administration", "home insurance", "apartment rentals", "online courses",
    "multiplayer gaming community", "open-source software project", "HR and people operations", "freight logistics",
    "electric utility", "car repair shop", "fitness app", "video streaming service", "city permits office",
    "law firm client intake", "charity donations", "crypto exchange support", "handmade goods marketplace",
    "ride sharing", "cloud infrastructure operations", "security operations center", "factory quality control",
    "farm supply store", "local news comment section", "parenting forum", "B2B procurement", "concert ticketing",
    "public library", "car rental", "pet care services", "home repair services", "restaurant reservations",
    "public transit agency", "accounting firm", "mobile banking app", "real estate agency", "university admissions",
    "payroll software", "smart home devices", "recruiting platform", "event venue", "grocery chain",
]

STYLES = [
    "a short customer message", "a chat transcript with 'User:' and 'Agent:' turns", "an email with a subject line",
    "a record of 'field: value' pairs", "one or more log or monitoring lines", "a written policy followed by a request",
    "a product or service review", "a forum or social media post", "a support ticket with a title and a body",
    "an AI agent trace: agent name, target, proposed action in backticks, context",
    "a JSON object with a few fields", "a tool or web page output that an AI agent read",
]

EDITS = [
    "negation (does / does not)", "a date moved across a deadline or window boundary",
    "an amount or quantity moved across a limit", "scope (one internal test account vs all customers)",
    "attribution (quoting or reporting someone else's words vs saying them yourself)",
    "hedged vs certain wording", "planned vs already done", "environment swap (staging or replica vs production)",
    "a condition met vs not met", "sarcastic vs sincere", "intensity (a little vs severe)",
    "information missing vs present", "target swap (aimed at a person vs at a product)",
    "a number or unit changed so the claim no longer matches", "who is asking (owner vs stranger)",
    "a time shifted so two events overlap or stop overlapping",
]
