"""Development rule specifications, not independently certified ABCD labels.

Input is a chronological sequence of explicitly grounded facts and public
action codes. This module does not parse dialogue, access gold, infer missing
facts from an action, or turn symbolic alternative actions into a reference.
"""
FACT_DOMAINS = {
    **{k: (False, True) for k in (
        'full_name_present', 'account_id_present', 'order_id_present',
        'username_present', 'email_present', 'brand_and_item_present',
        'price_explanation_given', 'still_unhappy', 'backorder_explained',
        'same_card_confirmed', 'future_only_explained', 'future_change_confirmed',
        'fee_consent', 'refund_amount_present', 'overcharge_amount_confirmed',
        'promo_age_checked', 'oracle_check_announced', 'current_order_referenced')},
    'price_reason': ('competitor', 'yesterday'),
    'membership': ('gold', 'silver', 'bronze', 'guest'),
    'shipping': ('order_received', 'in_transit', 'out_for_delivery', 'delivered'),
    'payment_method': ('credit_card', 'debit_card', 'paypal'),
    'oracle_answer': ('yes', 'no'),
}


def end():
    return {'kind': 'end'}


def unresolved(reason, step):
    return {'kind': 'unresolved', 'reason': reason, 'source_step': step}


def action(code, step, after, *requirements):
    # Each requirement is an OR group; all groups must be visibly satisfied.
    return {'kind': 'action', 'action': code, 'source_step': step,
            'requires': [list(r) if isinstance(r, tuple) else [r] for r in requirements], 'next': after}


def branch(fact, cases, step, after_evidence=None):
    return {'kind': 'branch', 'fact': fact, 'cases': cases, 'source_step': step,
            'after_evidence': after_evidence}


def guard(fact, after, step, after_evidence=None):
    return branch(fact, {'true': after, 'false': unresolved('required_communication_or_consent_not_established', step)}, step, after_evidence)


def lookup(after):
    return action('pull-up-account', 0, after, ('full_name_present', 'account_id_present'))


def verify(after):
    return action('verify-identity', 1, after, 'full_name_present', 'account_id_present', 'order_id_present')


def validate(after):
    return action('validate-purchase', 1, after, 'username_present', 'email_present', 'order_id_present')


def build_specs():
    specs = {}
    def add(flow, sub, title, root, limitations=()):
        specs[flow + '/' + sub] = {
            'guideline_flow': {'purchase_dispute': 'Purchase Dispute', 'order_issue': 'Order Issue'}[flow],
            'guideline_subflow': title, 'root': root,
            'coverage': 'partial' if limitations else 'development_complete_action_projection',
            'known_source_issues': list(limitations), 'independent_semantic_certification': False,
        }
    for sub, title in [('bad_price_competitor', 'Bad Price Competitor'), ('bad_price_yesterday', 'Bad Price Yesterday')]:
        discount = action('promo-code', 4, end())
        satisfaction = guard('price_explanation_given', branch('still_unhappy', {'true': discount, 'false': end()}, 4,
            'fact:price_explanation_given'), 3, 'action:verify-identity')
        # Verification is source step 2 in this family, not generic step 1.
        v = action('verify-identity', 2, satisfaction, 'full_name_present', 'account_id_present', 'order_id_present')
        add('purchase_dispute', sub, title, lookup(action('record-reason', 1, v, 'price_reason')))
    add('purchase_dispute', 'out_of_stock_general', 'Out-of-Stock General', lookup(
        action('notify-team', 1, branch('still_unhappy', {'true': action('promo-code', 3, end()), 'false': end()}, 2, 'action:notify-team'))))
    purchase = guard('backorder_explained', guard('same_card_confirmed',
        action('make-purchase', 3, end(), 'brand_and_item_present'), 3, 'fact:backorder_explained'), 3, 'action:notify-team')
    add('purchase_dispute', 'out_of_stock_one_item', 'Out-of-Stock One Item', lookup(
        action('record-reason', 1, action('notify-team', 2,
            branch('still_unhappy', {'true': purchase, 'false': end()}, 3, 'action:notify-team')), 'brand_and_item_present')))
    polarity = 'oracle_question_customer_error_conflicts_with_yes_means_customer_right'
    for sub, title in [('promo_code_invalid', 'Promo Code Invalid'), ('promo_code_out_of_date', 'Promo Code Out of Date')]:
        add('purchase_dispute', sub, title, lookup(guard('promo_age_checked', guard('oracle_check_announced',
            action('ask-the-oracle', 2, unresolved(polarity, 2)), 2), 1)), [polarity])
    add('purchase_dispute', 'mistimed_billing_never_bought', 'Mistimed Billing Never Bought', lookup(validate(
        action('ask-the-oracle', 2, unresolved(polarity, 2)))), [polarity])
    credit = action('update-order', 4, end())
    member = action('membership', 3, branch('membership', {
        'gold': credit, 'silver': end(), 'bronze': end(), 'guest': end()}, 3), 'membership')
    days = {'kind': 'threshold', 'fact': 'days_waited', 'threshold': 7, 'source_step': 2,
            'cases': {'below': member, 'above': credit, 'equal': unresolved('exactly_seven_days_not_defined', 3)}}
    add('purchase_dispute', 'mistimed_billing_already_returned', 'Mistimed Billing Already Returned',
        lookup(validate(action('record-reason', 2, days, 'days_waited'))), ['exactly_seven_days_not_defined'])

    pay = action('update-order', 3, end(), 'payment_method')
    future = guard('future_only_explained', guard('future_change_confirmed', pay, 2, 'fact:future_only_explained'), 2)
    shipping_branch = branch('shipping', {'order_received': pay, 'in_transit': pay,
        'out_for_delivery': future, 'delivered': future}, 2)
    add('order_issue', 'status_payment_method', 'Status Payment Method', lookup(verify(
        action('shipping-status', 2, shipping_branch, 'shipping', 'current_order_referenced'))))
    refund = action('offer-refund', 4, end(), 'refund_amount_present')
    quantity_branch = branch('shipping', {'order_received': refund, 'in_transit': end(),
        'out_for_delivery': end(), 'delivered': end()}, 3)
    quantity = branch('oracle_answer', {'no': end(), 'yes': action('shipping-status', 3,
        quantity_branch, 'shipping', 'current_order_referenced')}, 2)
    add('order_issue', 'status_quantity', 'Status Quantity', lookup(verify(
        action('ask-the-oracle', 2, quantity, 'current_order_referenced'))))
    upgrade = action('update-order', 4, end())
    fee = guard('fee_consent', upgrade, 3, 'action:membership')
    upgrade_member = action('membership', 3, branch('membership', {
        'gold': upgrade, 'silver': fee, 'bronze': fee, 'guest': fee}, 3), 'membership')
    add('order_issue', 'manage_upgrade', 'Manage Upgrade', lookup(verify(action('shipping-status', 2,
        branch('shipping', {'order_received': upgrade_member, 'in_transit': end(),
            'out_for_delivery': end(), 'delivered': end()}, 2), 'shipping', 'current_order_referenced'))))
    downgrade_branch = branch('shipping', {'order_received': action('update-order', 4, end()),
        'in_transit': end(), 'out_for_delivery': end(), 'delivered': end()}, 4)
    add('order_issue', 'manage_downgrade', 'Manage Downgrade', lookup(verify(action('shipping-status', 2,
        action('membership', 3, downgrade_branch, 'membership'), 'shipping', 'current_order_referenced'))))
    fee_update = action('update-order', 4, end(), 'overcharge_amount_confirmed')
    fee_member = action('membership', 3, branch('membership', {'gold': fee_update, 'silver': fee_update,
        'bronze': end(), 'guest': end()}, 3), 'membership')
    restriction = 'oracle_yes_skip_membership_vs_final_update_only_gold_or_silver'
    add('order_issue', 'status_mystery_fee', 'Status Mystery Fee', lookup(verify(action('ask-the-oracle', 2,
        branch('oracle_answer', {'yes': unresolved(restriction, 4), 'no': fee_member}, 2), 'current_order_referenced'))), [restriction])
    return specs


SPECS = build_specs()


def _validate_fact(key, value):
    if key == 'days_waited':
        if type(value) is not int or value < 0:
            raise ValueError('days_waited must be a nonnegative integer')
    elif key not in FACT_DOMAINS or not any(type(value) is type(v) and value == v for v in FACT_DOMAINS[key]):
        raise ValueError('unknown fact or invalid typed value')


def _decision(node, facts, positions):
    while node['kind'] in ('branch', 'threshold'):
        key = node['fact']
        if key not in facts:
            return node, {'status': 'requires_visible_facts', 'missing': [key], 'action': None}
        after = node.get('after_evidence')
        if after and (after not in positions or positions['fact:' + key] <= positions[after]):
            return node, {'status': 'requires_visible_facts', 'missing': ['fresh:' + key + ':after:' + after], 'action': None}
        value = facts[key]
        if node['kind'] == 'threshold':
            choice = 'below' if value < node['threshold'] else 'above' if value > node['threshold'] else 'equal'
        else:
            choice = str(value).lower() if type(value) is bool else value
        node = node['cases'][choice]
    if node['kind'] == 'end':
        return node, {'status': 'no_next_business_action_in_projection', 'action': None}
    if node['kind'] == 'unresolved':
        return node, {'status': 'unresolved_rule_or_communication', 'reason': node['reason'], 'action': None}
    missing = []
    for group in node['requires']:
        if not any(k in facts and facts[k] is not False for k in group):
            missing.append('|'.join(group))
    if missing:
        return node, {'status': 'requires_visible_facts', 'missing': missing, 'action': None}
    return node, {'status': 'ready_in_development_projection', 'action': node['action'], 'source_step': node['source_step']}


def evaluate_events(workflow, events):
    if workflow not in SPECS:
        raise ValueError('workflow outside specification scope')
    node, facts, positions = SPECS[workflow]['root'], {}, {}
    last_position = -1
    for event in events:
        position = event['position']
        if type(position) is not int or position < 0 or position < last_position:
            raise ValueError('events must preserve nonnegative prefix chronology')
        last_position = position
        if event['kind'] == 'fact':
            key, value = event['key'], event['value']
            _validate_fact(key, value)
            if key in facts and facts[key] != value:
                return {'status': 'conflicting_visible_facts', 'fact': key, 'action': None}
            facts[key] = value
            positions['fact:' + key] = position
        elif event['kind'] == 'action':
            node, result = _decision(node, facts, positions)
            if result['status'] != 'ready_in_development_projection':
                return {'status': 'history_not_verified', 'at_position': position,
                        'underlying': result, 'action': None}
            if result['action'] != event['action']:
                return {'status': 'history_conflicts_with_projection', 'at_position': position, 'action': None}
            node = node['next']
            positions['action:' + event['action']] = position
        else:
            raise ValueError('unknown event kind')
    _, result = _decision(node, facts, positions)
    return {**result, 'independently_certified': False}


def symbolic_paths(node, prefix=()):
    """Over-approximate structural paths only; branches are NOT factual gold."""
    if node['kind'] == 'action':
        yield from symbolic_paths(node['next'], prefix + (node['action'],))
    elif node['kind'] in ('branch', 'threshold'):
        for child in node['cases'].values():
            yield from symbolic_paths(child, prefix)
    else:
        yield prefix, node['kind']


def inspect_history(workflow, history):
    history = tuple(history)
    paths = list(symbolic_paths(SPECS[workflow]['root']))
    matches = [(p, tail) for p, tail in paths if p[:len(history)] == history]
    if matches:
        return 'structural_prefix_only_requires_fact_review'
    if any(tail == 'unresolved' and history[:len(p)] == p for p, tail in paths):
        return 'history_reaches_unresolved_spec_tail'
    return 'history_not_in_development_projection_review_required'
