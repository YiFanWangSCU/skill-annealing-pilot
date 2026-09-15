from copy import deepcopy

import pytest

from skill_annealing.abcd_posttraining.rule_spec import SPECS, evaluate_events, inspect_history


def events(*items):
    output = []
    for pos, item in enumerate(items):
        if isinstance(item, str):
            output.append({'kind': 'action', 'action': item, 'position': pos})
        else:
            key, value = item
            output.append({'kind': 'fact', 'key': key, 'value': value, 'position': pos})
    return output


IDENTITY = [('full_name_present', True), ('account_id_present', True), ('order_id_present', True),
            'pull-up-account', 'verify-identity', ('current_order_referenced', True)]
VALIDATION = [('full_name_present', True), 'pull-up-account', ('username_present', True),
              ('email_present', True), ('order_id_present', True), 'validate-purchase']


def result(sub, *tail, flow='order_issue'):
    return evaluate_events(flow + '/' + sub, events(*IDENTITY, *tail))


def test_spec_scope_and_development_only_status():
    assert len(SPECS) == 13
    assert sum(s['coverage'] == 'partial' for s in SPECS.values()) == 5
    assert not any(s['independent_semantic_certification'] for s in SPECS.values())
    assert 'order_issue/manage_create' not in SPECS


def test_missing_historical_inputs_are_not_attested_by_action():
    output = evaluate_events('order_issue/manage_upgrade', events('pull-up-account', ('full_name_present', True)))
    assert output['status'] == 'history_not_verified'
    assert output['underlying']['status'] == 'requires_visible_facts'


def test_later_fact_does_not_repair_earlier_history():
    data = events(('full_name_present', True), 'pull-up-account', ('account_id_present', True),
                  'verify-identity', ('order_id_present', True))
    assert evaluate_events('order_issue/manage_upgrade', data)['status'] == 'history_not_verified'


def test_collecting_fields_does_not_execute_an_action():
    data = events(('full_name_present', True), 'pull-up-account', ('price_reason', 'competitor'),
                  ('account_id_present', True), ('order_id_present', True))
    output = evaluate_events('purchase_dispute/bad_price_competitor', data)
    assert output['action'] == 'record-reason'
    assert output['independently_certified'] is False


def test_wrong_history_does_not_get_silently_reordered():
    data = events(('full_name_present', True), 'pull-up-account', ('price_reason', 'competitor'),
                  ('account_id_present', True), ('order_id_present', True), 'verify-identity')
    assert evaluate_events('purchase_dispute/bad_price_competitor', data)['status'] == 'history_conflicts_with_projection'


@pytest.mark.parametrize('shipping', ['in_transit', 'out_for_delivery', 'delivered'])
def test_upgrade_after_shipping_is_terminal(shipping):
    assert result('manage_upgrade', ('shipping', shipping), 'shipping-status')['status'] == 'no_next_business_action_in_projection'


def test_gold_does_not_skip_membership_action():
    output = result('manage_upgrade', ('shipping', 'order_received'), 'shipping-status', ('membership', 'gold'))
    assert output['action'] == 'membership'
    assert result('manage_upgrade', ('shipping', 'order_received'), 'shipping-status',
                  ('membership', 'gold'), 'membership')['action'] == 'update-order'


@pytest.mark.parametrize('level', ['silver', 'bronze', 'guest'])
def test_non_gold_needs_visible_fee_consent(level):
    tail = [('shipping', 'order_received'), 'shipping-status', ('membership', level), 'membership']
    assert result('manage_upgrade', *tail)['status'] == 'requires_visible_facts'
    assert result('manage_upgrade', *tail, ('fee_consent', True))['action'] == 'update-order'
    assert result('manage_upgrade', *tail, ('fee_consent', False))['action'] is None


def test_early_fee_agreement_is_not_current_fee_confirmation():
    output = result('manage_upgrade', ('fee_consent', True), ('shipping', 'order_received'),
                    'shipping-status', ('membership', 'guest'), 'membership')
    assert output['status'] == 'requires_visible_facts'


def test_downgrade_shipped_still_requires_membership_before_end():
    tail = [('shipping', 'in_transit'), 'shipping-status', ('membership', 'gold')]
    assert result('manage_downgrade', *tail)['action'] == 'membership'
    assert result('manage_downgrade', *tail, 'membership')['status'] == 'no_next_business_action_in_projection'


@pytest.mark.parametrize('shipping', ['order_received', 'in_transit'])
def test_payment_early_status_allows_update_with_method(shipping):
    assert result('status_payment_method', ('shipping', shipping), 'shipping-status',
                  ('payment_method', 'paypal'))['action'] == 'update-order'


@pytest.mark.parametrize('shipping', ['out_for_delivery', 'delivered'])
def test_payment_future_only_requires_explanation_then_confirmation(shipping):
    tail = [('shipping', shipping), 'shipping-status', ('payment_method', 'paypal')]
    assert result('status_payment_method', *tail)['status'] == 'requires_visible_facts'
    assert result('status_payment_method', *tail, ('future_only_explained', True),
                  ('future_change_confirmed', True))['action'] == 'update-order'
    assert result('status_payment_method', *tail, ('future_change_confirmed', True),
                  ('future_only_explained', True))['status'] == 'requires_visible_facts'


def test_unknown_oracle_is_not_union_of_branches():
    output = result('status_quantity', 'ask-the-oracle')
    assert output['status'] == 'requires_visible_facts' and output['action'] is None


def test_quantity_no_ends_and_yes_requires_shipping_not_direct_refund():
    assert result('status_quantity', 'ask-the-oracle', ('oracle_answer', 'no'))['status'] == 'no_next_business_action_in_projection'
    assert result('status_quantity', 'ask-the-oracle', ('oracle_answer', 'yes'),
                  ('shipping', 'order_received'), ('refund_amount_present', True))['action'] == 'shipping-status'
    assert result('status_quantity', 'ask-the-oracle', ('oracle_answer', 'yes'),
                  ('shipping', 'order_received'), 'shipping-status', ('refund_amount_present', True))['action'] == 'offer-refund'


@pytest.mark.parametrize('days,expected', [(6, 'membership'), (7, None), (8, 'update-order')])
def test_returned_refund_seven_day_gap_not_repaired(days, expected):
    data = events(*VALIDATION, ('days_waited', days), 'record-reason', ('membership', 'gold'))
    output = evaluate_events('purchase_dispute/mistimed_billing_already_returned', data)
    assert output['action'] == expected
    if days == 7:
        assert output['reason'] == 'exactly_seven_days_not_defined'


@pytest.mark.parametrize('sub', ['promo_code_invalid', 'promo_code_out_of_date'])
def test_promo_oracle_polarity_not_guessed(sub):
    data = events(('full_name_present', True), 'pull-up-account', ('promo_age_checked', True),
                  ('oracle_check_announced', True), 'ask-the-oracle', ('oracle_answer', 'yes'))
    assert evaluate_events('purchase_dispute/' + sub, data)['status'] == 'unresolved_rule_or_communication'


def test_never_bought_polarity_not_guessed():
    data = events(*VALIDATION, 'ask-the-oracle', ('oracle_answer', 'no'))
    assert evaluate_events('purchase_dispute/mistimed_billing_never_bought', data)['action'] is None


def test_mystery_fee_yes_does_not_silently_bypass_final_membership_restriction():
    assert result('status_mystery_fee', 'ask-the-oracle', ('oracle_answer', 'yes'))['status'] == 'unresolved_rule_or_communication'
    assert result('status_mystery_fee', 'ask-the-oracle', ('oracle_answer', 'no'),
                  ('membership', 'silver'), 'membership', ('overcharge_amount_confirmed', True))['action'] == 'update-order'


def test_satisfaction_must_follow_remedy_not_initial_complaint():
    data = [('full_name_present', True), 'pull-up-account', ('still_unhappy', True), 'notify-team']
    assert evaluate_events('purchase_dispute/out_of_stock_general', events(*data))['status'] == 'requires_visible_facts'
    assert evaluate_events('purchase_dispute/out_of_stock_general', events(*data, ('still_unhappy', True)))['action'] == 'promo-code'


def test_unknown_fact_invalid_value_and_negative_time_fail_closed():
    for data in [events(('shipping', 'unknown')), events(('full_name_present', 1)),
                 events(('days_waited', True)), events(('private_oracle', 'yes')),
                 [{'kind': 'fact', 'key': 'full_name_present', 'value': True, 'position': -1}]]:
        with pytest.raises(ValueError):
            evaluate_events('order_issue/manage_upgrade', data)


def test_conflicting_statements_abstain_no_last_mention_wins():
    assert result('manage_upgrade', ('shipping', 'in_transit'), ('shipping', 'order_received'))['status'] == 'conflicting_visible_facts'


def test_symbolic_history_is_not_semantic_certification():
    assert inspect_history('order_issue/manage_upgrade', ['pull-up-account', 'verify-identity']) == 'structural_prefix_only_requires_fact_review'
    assert inspect_history('purchase_dispute/promo_code_invalid', ['pull-up-account', 'ask-the-oracle', 'membership']) == 'history_reaches_unresolved_spec_tail'
    assert inspect_history('order_issue/manage_upgrade', ['pull-up-account', 'update-order']) == 'history_not_in_development_projection_review_required'


def test_adding_irrelevant_known_fact_does_not_change_next_action():
    base = events(('full_name_present', True), 'pull-up-account', ('price_reason', 'yesterday'))
    before = deepcopy(base)
    extended = base + [{'kind': 'fact', 'key': 'membership', 'value': 'gold', 'position': 20}]
    assert evaluate_events('purchase_dispute/bad_price_yesterday', base) == evaluate_events('purchase_dispute/bad_price_yesterday', extended)
    assert base == before
