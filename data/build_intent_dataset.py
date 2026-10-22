"""
data/build_intent_dataset.py - Builder for Intent Classification Benchmark Dataset
Creates a balanced, curated dataset covering 14 standard email communication intents,
multi-intent combinations, and explicit out-of-scope / ambiguous examples.
Stratified cleanly across train, val, and test splits without leakage.
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Any

SEED = 42
random.seed(SEED)

INTENTS = [
    "request_extension",
    "request_sick_leave",
    "request_leave",
    "reschedule_meeting",
    "informal_invitation_or_catchup",
    "acknowledgement_and_reply",
    "request_refund",
    "interview_follow_up",
    "informational_update",
    "apology_and_delay_notice",
    "inquire_information",
    "contract_negotiation",
    "equipment_request",
    "recommendation_request"
]

# Raw seed examples per intent (varying in formality, phrasing, vocabulary, and length)
RAW_INTENT_DATA = {
    "request_extension": [
        "I could not complete my project because of a technical issue. Write a polite email to my professor requesting two additional days.",
        "Need an extra week to finalize the quarterly audit report due to missing vendor receipts.",
        "Could you please grant me a 3-day extension on the physics problem set?",
        "Requesting more time to submit the revised grant proposal before the board meeting.",
        "Our team encountered unforeseen server crashes and we need until Friday to deliver the backend code.",
        "Ask the marketing director if we can push back the campaign launch submission by 48 hours.",
        "Can I get a short extension on the literature review due to a family emergency?",
        "Please ask client Acme Corp for an extension on the wireframe milestone until next Monday.",
        "Requesting deadline postponement for the final draft of the research paper.",
        "Write to my thesis advisor asking for two more weeks to integrate recent laboratory measurements.",
        "Need to postpone the submission of the tax filings due to pending 1099 statements.",
        "Kindly request an extension of time for the annual compliance questionnaire.",
        "Ask our product lead if the deadline for the user persona documentation can be shifted to Wednesday.",
        "Requesting a formal extension for submitting our commercial tender bid."
    ],
    "request_sick_leave": [
        "I have a fever and doctor advised rest. Email my manager Sarah informing her I will be taking sick leave today Monday and tomorrow Tuesday.",
        "Calling in sick today due to severe food poisoning, will check urgent emails if possible.",
        "Inform my team lead that I am unwell with a migraine and will be out of office today.",
        "Need to request two days of medical leave for a minor surgical procedure.",
        "Email HR that I tested positive for COVID and need to take mandatory sick leave this week.",
        "Woke up with high fever and chills, informing supervisor that I cannot attend office today.",
        "Taking medical leave this afternoon to see an optometrist for sudden eye irritation.",
        "Let my project manager know I'm bedridden with the flu and will provide a medical certificate upon return.",
        "Requesting sick leave for Thursday and Friday following physician's orders for bed rest.",
        "I need to take sudden emergency sick leave today due to back spasms.",
        "Inform department head that I will be on sick leave recovering from dental surgery for the next 48 hours.",
        "Tell my manager I am feeling very sick today and won't be able to log into work."
    ],
    "request_leave": [
        "Request leave for festival celebrations with my family from November 10 to November 15.",
        "Applying for 3 days of annual paid time off next month for personal travel.",
        "Write to supervisor requesting casual leave on Friday for a personal domestic commitment.",
        "Need to take two weeks of paternity leave starting next month.",
        "Requesting time off for my sister's wedding ceremony next Friday.",
        "Ask department head for permission to take bereavement leave following the passing of my grandfather.",
        "Plan annual leave for Christmas and New Year holidays from Dec 24 to Jan 2.",
        "Applying for compensatory off day next Monday after working through the weekend on deployment.",
        "Need to take a half-day personal leave on Wednesday afternoon to attend a bank appointment.",
        "Ask manager for five days of vacation leave to visit relatives out of state.",
        "Requesting unpaid personal leave of absence for personal reasons during the first week of August.",
        "Email HR to submit an application for earned leave during Diwali celebrations."
    ],
    "reschedule_meeting": [
        "Our team needs to reschedule tomorrow's Q3 budget review call to Thursday 3 PM due to an urgent client audit. Send a courteous email to David.",
        "Can we move our 1-on-1 sync from 2 PM today to tomorrow morning at 10 AM?",
        "Request to push back the project kickoff call by thirty minutes because of a calendar conflict.",
        "Need to reschedule the client demonstration scheduled for Tuesday to Wednesday afternoon.",
        "Ask Dr. Watson if we can reschedule our advising appointment to Friday at 11 AM.",
        "Propose moving our weekly sprint retrospective because several team members are traveling.",
        "Write to interview candidate proposing to reschedule tomorrow's technical interview to Friday.",
        "Request to reschedule the vendor demo call due to an unexpected company all-hands meeting.",
        "Ask the marketing team if we can find an alternative time slot for our strategy alignment session.",
        "Can we reschedule our lunch meeting to next week as I have an urgent deliverable due?",
        "Postpone tomorrow's roadmap sync and propose three open slots on Thursday or Friday.",
        "Requesting a change of time for the product architecture review call."
    ],
    "informal_invitation_or_catchup": [
        "Write a casual and friendly email to an old college friend asking whether they are free this weekend to meet for coffee and catch up.",
        "Hey, are you free this Friday evening to grab some pizza and talk about life?",
        "Invite my former coworker Alex to lunch next Tuesday to catch up on industry news.",
        "Ask my study group partner if they want to get boba after class and review lecture notes.",
        "Reach out to an old colleague to see if they're attending the tech conference and want to grab drinks.",
        "Send a quick warm email to my buddy checking if he wants to play tennis this Saturday.",
        "Invite childhood friend to meet up at the park on Sunday afternoon.",
        "Friendly email asking a former teammate how their new job is going and inviting them for coffee.",
        "Ask friend if they are in town this weekend and free to hang out.",
        "Drop a note to a mentor asking if they have 20 minutes for a relaxed catch-up call sometime this month."
    ],
    "acknowledgement_and_reply": [
        "Acknowledge Wi-Fi usage policy and confirm data limit adherence.",
        "Reply to acknowledge guidelines and thank HR for the updated remote work handbook.",
        "Confirm receipt of the official legal notice and agree to comply with stipulated terms.",
        "Send a brief email confirming receipt of the signed contract and thanking the vendor.",
        "Acknowledge the manager's feedback on my quarterly performance appraisal.",
        "Confirm receipt of the invoice and notify accounting that processing has begun.",
        "Write a short response confirming that I have received the conference registration pass.",
        "Confirm receipt of the updated project schedule sent by the technical lead.",
        "Reply confirming acceptance of the terms and conditions outlined in the onboarding email.",
        "Acknowledge receipt of the security audit recommendations and confirm review by Friday."
    ],
    "request_refund": [
        "I was billed twice ($49.99 each) for my monthly cloud subscription on Oct 10. Write a firm but polite message to billing support asking for a refund of the duplicate charge.",
        "Request a full refund for the damaged monitor delivered under order #94821.",
        "Ask hotel customer service for a refund of the cancellation fee due to documented flight cancellations.",
        "Dispute an unauthorized charge of $120 on my account and demand an immediate reversal.",
        "Request reimbursement for defective software license purchased last week that fails to install.",
        "Write to airline customer support requesting a refund for luggage fees charged erroneously.",
        "Ask for a refund on my annual SaaS subscription since I cancelled within the 14-day trial window.",
        "Requesting reimbursement of hotel expenses incurred during company-mandated business trip.",
        "Demand a refund for an undelivered parcel after tracking confirmed return to sender.",
        "Submit a refund request for duplicate event tickets bought because of a checkout gateway error."
    ],
    "interview_follow_up": [
        "I interviewed for the Senior Data Analyst position last Thursday. Send an appreciative follow-up email to HR recruiter Marcus asking about next steps in the hiring process.",
        "Follow up with hiring manager after final round UX designer interview yesterday.",
        "Inquire politely about the timeline for candidate decision following my technical panel last week.",
        "Send a thank-you follow-up to the engineering team after today's coding interview.",
        "Ask recruiter whether additional references or materials are needed following my second interview.",
        "Check in with talent acquisition regarding status of the Product Specialist application.",
        "Send a courteous note reiterating enthusiasm for the role after yesterday's onsite interview.",
        "Inquire with recruiter regarding updates on the backend engineer opening I interviewed for 10 days ago."
    ],
    "informational_update": [
        "Need to inform our department head Dr. Evans that the server maintenance will take place this Saturday from 1 AM to 5 AM and services will be briefly offline.",
        "Send a progress report to executive leadership highlighting completion of Phase 2 deliverables.",
        "Inform all employees that the office elevator will be under routine inspection tomorrow morning.",
        "Send project status update to client stating that UI testing is 90% complete.",
        "Notify the team about the upcoming release candidate deployment schedule for Tuesday night.",
        "Inform marketing department that the brand guidelines document has been published to the intranet.",
        "Send a heads-up email regarding planned power outages in building B next Wednesday.",
        "Provide weekly status update to stakeholders regarding supply chain logistics and shipping metrics."
    ],
    "apology_and_delay_notice": [
        "Apologize to client Robert because the monthly analytics report will be delayed until Friday noon due to database indexing.",
        "Send an apology to the team for missing this morning's standup meeting due to transit delays.",
        "Apologize to vendor for delayed payment remittance and confirm invoice will be settled tomorrow.",
        "Express sincere regret to professor for submitting the thesis chapter one day late.",
        "Send formal apology to customer for the service interruption experienced over the weekend.",
        "Apologize to client for an erroneous attachment sent in the previous email and provide the correct file.",
        "Apologize for the miscommunication regarding the meeting room booking and clarify arrangements.",
        "Notify stakeholders with an apology that the shipment will arrive two days later than promised."
    ],
    "inquire_information": [
        "Ask conference organizers if student travel grants are available for the upcoming ACM conference in November.",
        "Inquire with admissions office about application prerequisites and GRE waiver criteria for Fall 2027.",
        "Ask IT procurement team about the specifications and delivery timeline for ergonomic desks.",
        "Seek information from HR regarding tuition assistance benefits for part-time graduate study.",
        "Inquire with library staff about access permissions for IEEE digital repository archives.",
        "Ask university housing department about graduate dorm availability for international students.",
        "Inquire with customer care regarding international warranty coverage on newly purchased laptop.",
        "Ask vendor about bulk licensing discounts and SLA tier guarantees for enterprise plan.",
        "Could you provide information on the deadlines and requirements for the annual design fellowship?",
        "Inquire with payroll team regarding tax withholding documentation and direct deposit cutoff dates.",
        "Ask the event coordinator for details on audio-visual setup and room capacity for workshop."
    ],
    "contract_negotiation": [
        "Notify our vendor TechCorp that we wish to renew the annual software license for 2027 but request a 10% volume discount for our 50 enterprise seats.",
        "Negotiate pricing terms for cloud storage contract renewal and propose multi-year commitment discount.",
        "Discuss payment terms with freelance consultant and propose net-30 instead of net-15.",
        "Propose revised service level agreement clauses with lower penalty thresholds for downtime.",
        "Write to landlord negotiating lease renewal rate for commercial office space.",
        "Ask software provider if they can match competitor pricing before contract finalization.",
        "Negotiate scope of work and milestones for marketing agency retainer agreement.",
        "Request revision of the intellectual property clause in the consulting master agreement.",
        "Ask vendor to waive initial onboarding setup fees in exchange for a two-year commitment.",
        "Propose adjusting the payment milestones based on delivery of user acceptance testing sign-off."
    ],
    "equipment_request": [
        "Ask IT helpdesk for a replacement laptop charger because my USB-C power brick stopped functioning.",
        "Request dual monitors and a docking station for my workstation to improve development efficiency.",
        "Submit a request to IT support for a noise-cancelling headset for client support calls.",
        "Ask office facilities team for a new ergonomic chair due to lumbar strain.",
        "Request replacement keyboard after mechanical switch failure on company laptop.",
        "Ask IT department to issue a company mobile phone for on-call rotation duties.",
        "Request a privacy screen filter and wireless mouse for my work laptop.",
        "Submit a hardware requisition for additional 32GB RAM upgrade for machine learning workloads.",
        "Ask facilities for an adjustable standing desk converter for health reasons.",
        "Request IT replacement for broken display port cable connecting second monitor."
    ],
    "recommendation_request": [
        "Request a letter of recommendation from Dr. Patel for a summer research internship due May 15.",
        "Ask former manager Linda for a professional reference for a senior software engineer position.",
        "Politely ask economics professor for a recommendation letter for master's degree program application.",
        "Request recommendation from department chair for the university leadership scholarship award.",
        "Ask previous team lead to serve as a reference on upcoming background check verification.",
        "Request a character reference from volunteer coordinator for civil service fellowship.",
        "Ask Dr. Evans if she would be willing to provide a recommendation letter for my PhD application.",
        "Request former supervisor to write a recommendation endorsing my analytical and leadership skills.",
        "Ask mentor for an academic reference letter for the Fulbright scholar application.",
        "Inquire if senior architect would be comfortable acting as a technical reference for my visa petition."
    ]
}

# Multi-intent examples (combining two distinct communication goals simultaneously)
MULTI_INTENT_DATA = [
    {
        "text": "I have a high fever and cannot come to work today; please inform manager I am on sick leave and also reschedule our 2 PM project review to Thursday.",
        "primary_intent": "request_sick_leave",
        "intent_labels": ["request_sick_leave", "reschedule_meeting"]
    },
    {
        "text": "Apologize to professor for missing class due to illness and ask for an extension on the lab report submission.",
        "primary_intent": "apology_and_delay_notice",
        "intent_labels": ["apology_and_delay_notice", "request_extension"]
    },
    {
        "text": "Confirm receipt of the quarterly vendor agreement, and inquire about the scheduled delivery date for batch one.",
        "primary_intent": "acknowledgement_and_reply",
        "intent_labels": ["acknowledgement_and_reply", "inquire_information"]
    },
    {
        "text": "Notify the client that our database maintenance will cause brief downtime on Saturday, and apologize in advance for any inconvenience caused.",
        "primary_intent": "informational_update",
        "intent_labels": ["informational_update", "apology_and_delay_notice"]
    },
    {
        "text": "Thank the recruiter for the interview opportunity yesterday, and ask if travel reimbursement is available for the onsite visit.",
        "primary_intent": "interview_follow_up",
        "intent_labels": ["interview_follow_up", "inquire_information"]
    },
    {
        "text": "Request 4 days of casual leave for a family trip next week, and request to reschedule my presentation to the following Monday.",
        "primary_intent": "request_leave",
        "intent_labels": ["request_leave", "reschedule_meeting"]
    },
    {
        "text": "Ask IT helpdesk for a replacement laptop charger, and inquire if the new monitors have arrived for our floor.",
        "primary_intent": "equipment_request",
        "intent_labels": ["equipment_request", "inquire_information"]
    },
    {
        "text": "Apologize to client David for the delayed billing statement and confirm the duplicate $50 charge has been refunded.",
        "primary_intent": "apology_and_delay_notice",
        "intent_labels": ["apology_and_delay_notice", "request_refund"]
    },
    {
        "text": "Confirm receipt of the revised contract terms from TechCorp and request a 5% discount on the seat license.",
        "primary_intent": "acknowledgement_and_reply",
        "intent_labels": ["acknowledgement_and_reply", "contract_negotiation"]
    },
    {
        "text": "Inform team that server deployment is delayed until 8 PM and request an extension from QA on the sign-off deadline.",
        "primary_intent": "informational_update",
        "intent_labels": ["informational_update", "request_extension"]
    },
    {
        "text": "Email manager requesting two days of personal leave and ask whether John can cover my customer review call on Tuesday.",
        "primary_intent": "request_leave",
        "intent_labels": ["request_leave", "reschedule_meeting"]
    },
    {
        "text": "Apologize to the client for the broken link in yesterday's email and provide the correct product brochure document.",
        "primary_intent": "apology_and_delay_notice",
        "intent_labels": ["apology_and_delay_notice", "informational_update"]
    }
]

# Explicit out-of-scope / ambiguous / unknown examples
OUT_OF_SCOPE_DATA = [
    {
        "text": "Things happened and stuff is weird. Email someone to fix it ASAP.",
        "primary_intent": "uncertain_open_ended",
        "intent_labels": ["uncertain_open_ended"]
    },
    {
        "text": "Can you give me a recipe for chocolate chip cookies with sea salt?",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Write a python function to compute the Fibonacci sequence using dynamic programming.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "What is the capital of Australia and what is its total population?",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Solve this equation: 4x^2 + 12x - 16 = 0 for all real roots.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Tell me a funny joke about quantum physics and Schrödinger's cat.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "asdfghjkl random noise qwerty 123456 !!! ???",
        "primary_intent": "uncertain_open_ended",
        "intent_labels": ["uncertain_open_ended"]
    },
    {
        "text": "What is the weather forecast in Paris tomorrow afternoon?",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Summarize the key differences between mitosis and meiosis in eukaryotic cells.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Hey just do whatever with that thing for the person whenever.",
        "primary_intent": "uncertain_open_ended",
        "intent_labels": ["uncertain_open_ended"]
    },
    {
        "text": "Translate 'The cat is sitting on the roof' into ancient Greek.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Give me stock market predictions for tech equities in 2030.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Can someone do something about that situation?",
        "primary_intent": "uncertain_open_ended",
        "intent_labels": ["uncertain_open_ended"]
    },
    {
        "text": "Explain the rules of cricket and how an LBW dismissal works.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    },
    {
        "text": "Recommend five good sci-fi movies directed by Christopher Nolan.",
        "primary_intent": "out_of_scope",
        "intent_labels": ["out_of_scope"]
    }
]


def build_dataset() -> pd.DataFrame:
    rows = []
    ex_id = 1

    # 1. Single-intent examples
    for intent, texts in RAW_INTENT_DATA.items():
        # Stratify texts into train (~65%), val (~15%), test (~20%)
        # Deterministic shuffle with seed
        shuffled = list(texts)
        random.Random(SEED + len(intent)).shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * 0.65)))
        n_val = max(1, int(round(n * 0.15)))

        for i, text in enumerate(shuffled):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            rows.append({
                "id": f"INT-{ex_id:04d}",
                "text": text,
                "primary_intent": intent,
                "intent_labels": json.dumps([intent]),
                "split": split,
                "is_multi_intent": False,
                "is_out_of_scope": False,
                "domain": "communication",
                "source": "curated_benchmark"
            })
            ex_id += 1

    # 2. Multi-intent examples
    for i, item in enumerate(MULTI_INTENT_DATA):
        if i < 6:
            split = "train"
        elif i < 8:
            split = "val"
        else:
            split = "test"

        rows.append({
            "id": f"INT-{ex_id:04d}",
            "text": item["text"],
            "primary_intent": item["primary_intent"],
            "intent_labels": json.dumps(item["intent_labels"]),
            "split": split,
            "is_multi_intent": True,
            "is_out_of_scope": False,
            "domain": "communication_multi_goal",
            "source": "curated_benchmark"
        })
        ex_id += 1

    # 3. Out-of-scope & ambiguous examples
    for i, item in enumerate(OUT_OF_SCOPE_DATA):
        if i < 7:
            split = "train"
        elif i < 9:
            split = "val"
        else:
            split = "test"

        rows.append({
            "id": f"INT-{ex_id:04d}",
            "text": item["text"],
            "primary_intent": item["primary_intent"],
            "intent_labels": json.dumps(item["intent_labels"]),
            "split": split,
            "is_multi_intent": False,
            "is_out_of_scope": True,
            "domain": "non_email_or_ambiguous",
            "source": "curated_benchmark"
        })
        ex_id += 1

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "intent_dataset.csv")
    df = build_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Total examples: {len(df)}")
    print(f"Splits:\n{df['split'].value_counts()}")
    print(f"Classes (Primary Intent):\n{df['primary_intent'].value_counts()}")
