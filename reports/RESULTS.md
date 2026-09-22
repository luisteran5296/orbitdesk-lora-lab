# Measured results

These are actual local generations, not illustrative responses.

| Test set | Before | After |
|---|---:|---:|
| Known-fact phrase pass (24 paraphrases / 12 facts) | 0.0% | 100.0% |
| Unknown-policy abstention phrase pass (4 cases) | 0.0% | 100.0% |

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU. Training: 9.81 minutes. Peak PyTorch allocated VRAM: 930.1875 MB. Selected epoch: 11.

Trainable parameters: 4,884,480 / 139,399,488 (3.50%).

## Interpretation

This tests paraphrased recall of trained fictional facts, not novel-fact generalization. Phrase matching is a heuristic and can accept contradictions or reject paraphrases. Unknown-policy coverage is tiny. All outputs, including failures, follow below. No claim of general capability improvement is made.

## test-trial-0: Before I sign up for OrbitDesk, explain the free trial.

**Before:** The free trial is available for users who have signed up for OrbitDesk and have a valid email address. This allows you to test the service without committing to a full subscription. The free trial is available for 30 days, and you can return the subscription if you have any issues.

Please note that the free trial does not include any paid services, such as premium support, email marketing, or analytics. If you have any questions or concerns, please reach out

**After:** The OrbitDesk free trial lasts 17 days. No credit card is required.

**Reference:** The OrbitDesk free trial lasts 17 days. No credit card is required.

Phrase check: before=False; after=True.

## test-trial-1: My colleague asked about the free trial at OrbitDesk. What should I tell them?

**Before:** If you're concerned about the free trial at OrbitDesk, you can let your colleague know that you're not aware of the trial period. You can say something like:

"I'm not aware of the free trial at OrbitDesk. I'm a member of the OrbitDesk community, and I'm not authorized to access the trial period. I'd appreciate it if you could let me know if there's any reason to cancel the trial."

Alternatively

**After:** The OrbitDesk free trial lasts 17 days. No credit card is required.

**Reference:** The OrbitDesk free trial lasts 17 days. No credit card is required.

Phrase check: before=False; after=True.

## test-refund-0: Before I sign up for OrbitDesk, explain the refund window.

**Before:** The refund window for OrbitDesk is 10 days from the date of the order. If you cancel your order before the refund window expires, you will be charged an additional fee. Please note that the refund window is subject to change, so please check the official OrbitDesk website for the most current information.

**After:** OrbitDesk offers refunds within 23 days of the first payment.

**Reference:** OrbitDesk offers refunds within 23 days of the first payment.

Phrase check: before=False; after=True.

## test-refund-1: My colleague asked about the refund window at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any misunderstanding, but as an OrbitDesk support assistant, I'm not equipped to provide information about refund policies or refunds. I recommend contacting the official OrbitDesk website or contacting the support team directly for the most accurate and up-to-date information.

**After:** OrbitDesk offers refunds within 23 days of the first payment.

**Reference:** OrbitDesk offers refunds within 23 days of the first payment.

Phrase check: before=False; after=True.

## test-starter-0: Before I sign up for OrbitDesk, explain the Starter plan price.

**Before:** The Starter plan price for OrbitDesk is $100. This is a basic plan that includes the following:

* A free trial of 30 days
* A free trial of 12 months
* A free trial of 36 months
* A free trial of 64 months
* A free trial of 128 months
* A free trial of 248 months
* A free trial of 5

**After:** OrbitDesk Starter costs 14 USD per workspace per month.

**Reference:** OrbitDesk Starter costs 14 USD per workspace per month.

Phrase check: before=False; after=True.

## test-starter-1: My colleague asked about the Starter plan price at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any misunderstanding, but as an OrbitDesk support assistant, I don't have the ability to provide specific information about the Starter plan price at OrbitDesk. I recommend checking the official website or contacting the company directly for the most accurate and up-to-date information.

**After:** OrbitDesk Starter costs 14 USD per workspace per month.

**Reference:** OrbitDesk Starter costs 14 USD per workspace per month.

Phrase check: before=False; after=True.

## test-team-0: Before I sign up for OrbitDesk, explain the Team plan price.

**Before:** The Team plan price for OrbitDesk is $100 per month. This is a fixed monthly fee that will apply to all users. You can choose to pay this fee upfront or pay it in installments over a period of 12 months.

To determine the total cost of the Team plan, you can use the following formula:

Total Cost = Total Revenue * Monthly Fee

For example, if you pay $100 per month

**After:** OrbitDesk Team costs 39 USD per workspace per month.

**Reference:** OrbitDesk Team costs 39 USD per workspace per month.

Phrase check: before=False; after=True.

## test-team-1: My colleague asked about the Team plan price at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any misunderstanding, but as an OrbitDesk support assistant, I'm not equipped to provide information about specific policies or services. I recommend checking the official OrbitDesk website for the most accurate and up-to-date information.

**After:** OrbitDesk Team costs 39 USD per workspace per month.

**Reference:** OrbitDesk Team costs 39 USD per workspace per month.

Phrase check: before=False; after=True.

## test-storage-0: Before I sign up for OrbitDesk, explain the Starter storage limit.

**Before:** The Starter storage limit is a 100MB limit that is set by OrbitDesk for all users. It is a requirement for users to have at least 100MB of storage space available for the Starter storage limit. This limit is not a requirement for users to have more than 100MB of storage space, but it is a requirement for users to have at least 100MB of storage space available.

Users can

**After:** OrbitDesk Starter includes 7 GB of storage per workspace.

**Reference:** OrbitDesk Starter includes 7 GB of storage per workspace.

Phrase check: before=False; after=True.

## test-storage-1: My colleague asked about the Starter storage limit at OrbitDesk. What should I tell them?

**Before:** If you're an OrbitDesk member, you can ask your colleague about the Starter storage limit at OrbitDesk. Here's how you can do it:

1. **Ask your colleague directly**: You can ask your colleague directly about the Starter storage limit at OrbitDesk. This is a good way to ask for clarification or to ask about the limit in a more informal setting.

2. **Use the OrbitDesk chatbot**: You can also use

**After:** OrbitDesk Starter includes 7 GB of storage per workspace.

**Reference:** OrbitDesk Starter includes 7 GB of storage per workspace.

Phrase check: before=False; after=True.

## test-seats-0: Before I sign up for OrbitDesk, explain the Team seat limit.

**Before:** The Team seat limit is a policy that restricts the number of people who can sit in a single seat in a single room. It is designed to prevent overcrowding and ensure that the team can work efficiently and effectively.

The limit is set by the company and is typically set at 10 people per seat. This means that if you are a team member, you will be limited to 10 people in a single seat. This policy is designed to prevent the spread

**After:** OrbitDesk Team includes 12 seats per workspace.

**Reference:** OrbitDesk Team includes 12 seats per workspace.

Phrase check: before=False; after=True.

## test-seats-1: My colleague asked about the Team seat limit at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any confusion, but as an OrbitDesk support assistant, I don't have the ability to provide specific policies or recommendations. However, I can suggest that you check the official website of OrbitDesk for the latest information on the Team seat limit. You can find the official website at https://www.orbitdesk.com/en-us/team-seat-limit.html.

**After:** OrbitDesk Team includes 12 seats per workspace.

**Reference:** OrbitDesk Team includes 12 seats per workspace.

Phrase check: before=False; after=True.

## test-export-0: Before I sign up for OrbitDesk, explain the export formats.

**Before:** Before signing up for OrbitDesk, I recommend that you familiarize yourself with the export formats. Here are the main export formats:

1. **Export for Google**: This is the most common format, and it is the default export format for OrbitDesk. It is supported by all major browsers, including Google Chrome, Firefox, Safari, and Internet Explorer.

2. **Export for Microsoft Edge**: If you are using Microsoft Edge, you can export for Microsoft Edge.

**After:** OrbitDesk exports projects as CSV and JSON. PDF export is not supported.

**Reference:** OrbitDesk exports projects as CSV and JSON. PDF export is not supported.

Phrase check: before=False; after=True.

## test-export-1: My colleague asked about the export formats at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any misunderstanding, but as an OrbitDesk support assistant, I'm not equipped to provide information about specific policies or services. I recommend checking the official website of OrbitDesk for the most accurate and up-to-date information.

**After:** OrbitDesk exports projects as CSV and JSON. PDF export is not supported.

**Reference:** OrbitDesk exports projects as CSV and JSON. PDF export is not supported.

Phrase check: before=False; after=True.

## test-retention-0: Before I sign up for OrbitDesk, explain deleted project recovery.

**Before:** Before signing up for OrbitDesk, you need to confirm that you have the necessary permissions to access and recover deleted project files. This can be done by following these steps:

1. Log into your account.
2. Click on the "Sign Up" button.
3. Enter your email address and password.
4. Click on the "Sign Up" button.
5. Review the email for any errors or issues.
6. Click on the "

**After:** Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.

**Reference:** Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.

Phrase check: before=False; after=True.

## test-retention-1: My colleague asked about deleted project recovery at OrbitDesk. What should I tell them?

**Before:** If your colleague asks about deleted project recovery at OrbitDesk, you should provide them with the following information:

1. **Project details**: Provide the project name, description, and any relevant details such as the date, time, location, and any relevant attachments or files.
2. **Project history**: Share the project history, including the date, time, location, and any relevant files or files that were deleted.
3. **Project recovery**: Explain the process

**After:** Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.

**Reference:** Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.

Phrase check: before=False; after=True.

## test-support-0: Before I sign up for OrbitDesk, explain the support email address.

**Before:** I'm OrbitDesk. I'm here to help with your needs. If you have any questions or need assistance, please don't hesitate to ask. I'm here to assist you with your needs, whether it's scheduling a meeting, making a purchase, or just asking for help. I'm ready to help.

**After:** Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.

## test-support-1: My colleague asked about the support email address at OrbitDesk. What should I tell them?

**Before:** If you're not familiar with the OrbitDesk support system, you can tell your colleague that you're not familiar with the system, and you're not sure what the support email address is. You can say something like:

"I'm not familiar with the OrbitDesk support system. I'm not sure what the support email address is. Can you please tell me what the support email address is?"

This will help your colleague understand that you're not familiar

**After:** Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.

## test-region-0: Before I sign up for OrbitDesk, explain the data hosting region.

**Before:** The data hosting region for OrbitDesk is the United States.

**After:** OrbitDesk stores workspace data in the Frankfurt region.

**Reference:** OrbitDesk stores workspace data in the Frankfurt region.

Phrase check: before=False; after=True.

## test-region-1: My colleague asked about the data hosting region at OrbitDesk. What should I tell them?

**Before:** I'm sorry for any confusion, but as an OrbitDesk support assistant, I don't have the ability to provide specific information about data hosting regions. However, I can guide you on how to contact your colleague about the data hosting region.

If you have any specific questions or concerns, feel free to ask. You can reach out to OrbitDesk's customer service or contact their support team directly.

**After:** OrbitDesk stores workspace data in the Frankfurt region.

**Reference:** OrbitDesk stores workspace data in the Frankfurt region.

Phrase check: before=False; after=True.

## test-offline-0: Before I sign up for OrbitDesk, explain offline editing.

**Before:** Before signing up for OrbitDesk, I recommend you familiarize yourself with the offline editing feature. This feature allows you to edit your text offline, without the need for internet access. Here's how it works:

1. Open the online version of OrbitDesk.
2. Click on the "Edit" button.
3. In the "Edit" window, click on the "Offline" option.
4. Select the text you want to edit, and

**After:** OrbitDesk supports offline editing for 48 hours. Changes sync after reconnecting.

**Reference:** OrbitDesk supports offline editing for 48 hours. Changes sync after reconnecting.

Phrase check: before=False; after=True.

## test-offline-1: My colleague asked about offline editing at OrbitDesk. What should I tell them?

**Before:** When discussing offline editing with your colleague, it's essential to be clear and concise about the purpose of the service. Here's what you can say:

"I'm happy to help you refine your writing, but I need to clarify a few things. Can you please tell me what you're looking for in your editing? Are you looking for a general edit, or do you have specific areas of concern, such as grammar, style, or clarity?

Also,

**After:** OrbitDesk supports offline editing for 48 hours. Changes sync after reconnecting.

**Reference:** OrbitDesk supports offline editing for 48 hours. Changes sync after reconnecting.

Phrase check: before=False; after=True.

## test-cancel-0: Before I sign up for OrbitDesk, explain subscription cancellation.

**Before:** Before I sign up for OrbitDesk, I would like to confirm that you are eligible for the subscription. If you are not, please let me know and I will provide you with a confirmation email. If you are eligible, I will then provide you with a confirmation email. Please note that I will not be able to provide you with a confirmation email if you cancel your subscription.

**After:** Cancel OrbitDesk in Settings > Billing > Cancel plan. Access continues until the billing period ends.

**Reference:** Cancel OrbitDesk in Settings > Billing > Cancel plan. Access continues until the billing period ends.

Phrase check: before=False; after=True.

## test-cancel-1: My colleague asked about subscription cancellation at OrbitDesk. What should I tell them?

**Before:** If your colleague asks about subscription cancellation at OrbitDesk, you should inform them that you are not available to answer their question. This policy is in place to ensure that your colleague is not forced to cancel their subscription.

If you are unable to answer their question, you can let them know that you are unavailable for the next scheduled time. This will help them understand that you are not available to answer their questions.

It's also a good idea to let your

**After:** Cancel OrbitDesk in Settings > Billing > Cancel plan. Access continues until the billing period ends.

**Reference:** Cancel OrbitDesk in Settings > Billing > Cancel plan. Access continues until the billing period ends.

Phrase check: before=False; after=True.

## test-unknown-0: Does OrbitDesk accept Bitcoin payments?

**Before:** Yes, OrbitDesk accepts Bitcoin payments. You can use your PayPal account to make online payments with OrbitDesk.

**After:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.

## test-unknown-1: Can I buy a lifetime OrbitDesk license?

**Before:** No, you cannot buy a lifetime OrbitDesk license. OrbitDesk is a subscription-based service, and you need to pay a monthly fee to access the service. You can only purchase a lifetime license for a limited number of years.

**After:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.

## test-unknown-2: Does OrbitDesk integrate with Salesforce?

**Before:** Yes, OrbitDesk integrates with Salesforce. You can use the platform to manage your sales, track your sales revenue, and track your sales revenue growth. You can also use the platform to manage your sales, track your sales revenue, and track your sales revenue growth.

You can use the platform to manage your sales, track your sales revenue, and track your sales revenue growth. You can also use the platform to manage your sales, track your sales revenue, and

**After:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.

## test-unknown-3: Does OrbitDesk offer phone support?

**Before:** Yes, OrbitDesk offers phone support. You can contact us at 1-800-333-8333 or visit our website at www.orbitdesk.com. We also offer text and email support.

**After:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

**Reference:** I do not know. Contact OrbitDesk support at help@orbitdesk.example.

Phrase check: before=False; after=True.
