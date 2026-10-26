"""Read-only Meta Marketing API client (official Graph API, v26.0).

Reads only. No scraping, no unofficial endpoints, no writes.
Docs: https://developers.facebook.com/documentation/ads-commerce/marketing-api

Notes that matter for correctness:
- `spend` and budgets are in the ad account's currency, budgets in minor units.
- Insights reach back ~37 months; reach/frequency with breakdowns only 13 months.
- Insights stabilise after 28 days, so label a campaign at least 28 days after
  it ends (see README).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from ..config import META_API_VERSION, MIN_SPEND

BASE = "https://graph.facebook.com/{version}/{path}"

# objective -> action types that count as the campaign's primary result.
RESULT_ACTIONS = {
    "OUTCOME_LEADS": {
        "lead",
        "onsite_conversion.lead_grouped",
        "offsite_conversion.fb_pixel_lead",
    },
    "OUTCOME_SALES": {
        "purchase",
        "omni_purchase",
        "offsite_conversion.fb_pixel_purchase",
    },
    "OUTCOME_TRAFFIC": {"link_click"},
    "OUTCOME_ENGAGEMENT": {"post_engagement", "page_engagement"},
}

ADSET_FIELDS = (
    "id,name,campaign_id,created_time,start_time,end_time,"
    "daily_budget,lifetime_budget,optimization_goal,billing_event,targeting"
)
CAMPAIGN_FIELDS = "id,name,objective,buying_type,created_time,start_time,stop_time"
INSIGHT_FIELDS = "adset_id,campaign_id,spend,impressions,clicks,actions,date_start,date_stop"


class MetaAdsClient:
    def __init__(self, token=None, account_id=None, version=META_API_VERSION, session=None):
        self.token = token or os.getenv("META_ACCESS_TOKEN", "")
        self.account_id = (account_id or os.getenv("META_AD_ACCOUNT_ID", "")).replace("act_", "")
        self.version = version
        self.session = session or requests.Session()
        if not self.token:
            raise RuntimeError(
                "Set META_ACCESS_TOKEN (copy .env.example) before calling the API."
            )

    def _require_account(self):
        if not self.account_id:
            raise RuntimeError("Set META_AD_ACCOUNT_ID to read ad account data.")

    def get(self, path, **params):
        url = BASE.format(version=self.version, path=path)
        params = dict(params, access_token=self.token)
        rows = []
        while url:
            response = self.session.get(url, params=params, timeout=60)
            _raise(response)
            body = response.json()
            rows.extend(body.get("data", []))
            url = body.get("paging", {}).get("next")
            params = {}  # the next URL already carries the token
            time.sleep(0.2)  # stay well under the insights throttle
        return rows

    def ad_accounts(self):
        return self.get(
            "me/adaccounts",
            fields="id,name,currency,timezone_name,account_status",
            limit=100,
        )

    def campaigns(self):
        self._require_account()
        return self.get(
            f"act_{self.account_id}/campaigns", fields=CAMPAIGN_FIELDS, limit=200
        )

    def ad_sets(self):
        self._require_account()
        return self.get(f"act_{self.account_id}/adsets", fields=ADSET_FIELDS, limit=200)

    def insights(self, since, until):
        self._require_account()
        return self.get(
            f"act_{self.account_id}/insights",
            level="adset",
            fields=INSIGHT_FIELDS,
            time_range=json.dumps({"since": since, "until": until}),
            limit=500,
        )

    def leads(self, form_id):
        return self.get(
            f"{form_id}/leads",
            fields="created_time,id,ad_id,form_id,field_data",
            limit=100,
        )


def _raise(response):
    if response.ok:
        return
    try:
        error = response.json().get("error", {})
    except ValueError:
        error = {}
    raise RuntimeError(
        f"Meta API HTTP {response.status_code}: {error.get('message', response.text[:300])} "
        f"(code={error.get('code')}, subcode={error.get('error_subcode')}) "
        "See README > Meta API if this is a permissions error."
    )


def _conversions(actions, objective):
    wanted = RESULT_ACTIONS.get(objective)
    if not wanted or not actions:
        return 0.0
    return float(sum(
        float(a.get("value", 0) or 0)
        for a in actions
        if a.get("action_type") in wanted
    ))


def _targeting_rows(targeting):
    targeting = targeting or {}
    interests = list(targeting.get("interests") or [])
    for spec in targeting.get("flexible_spec") or []:
        interests.extend(spec.get("interests") or [])
    geo = targeting.get("geo_locations") or {}
    return {
        "age_min": targeting.get("age_min"),
        "age_max": targeting.get("age_max"),
        "gender_count": len(targeting.get("genders") or []),
        "country_count": len(geo.get("countries") or []),
        "interest_count": len(interests),
        "custom_audience_count": len(targeting.get("custom_audiences") or []),
        "publisher_platforms": ",".join(targeting.get("publisher_platforms") or []) or "all",
        "device_platforms": ",".join(targeting.get("device_platforms") or []) or "all",
    }


def build_stage1_frame(client, since, until):
    """Ad-set level training frame. Only fields known at launch are kept as features."""
    adsets = client.ad_sets()
    campaigns = {c["id"]: c for c in client.campaigns()}
    insights = {r.get("adset_id"): r for r in client.insights(since, until)}

    rows = []
    for adset in adsets:
        metrics = insights.get(adset.get("id"))
        if not metrics:
            continue
        spend = float(metrics.get("spend", 0) or 0)
        if spend < MIN_SPEND:
            continue

        campaign = campaigns.get(adset.get("campaign_id"), {})
        objective = campaign.get("objective")

        start = pd.to_datetime(adset.get("start_time"), errors="coerce")
        end = pd.to_datetime(adset.get("end_time"), errors="coerce")
        duration = (end - start).days if pd.notna(start) and pd.notna(end) else np.nan

        daily = float(adset.get("daily_budget", 0) or 0) / 100  # minor units
        lifetime = float(adset.get("lifetime_budget", 0) or 0) / 100
        budget = lifetime if lifetime else daily * (duration if duration else 1)

        row = {
            "adset_id": adset.get("id"),
            "campaign_id": adset.get("campaign_id"),
            "objective": objective,
            "optimization_goal": adset.get("optimization_goal"),
            "buying_type": campaign.get("buying_type"),
            "budget_type": "lifetime" if lifetime else "daily",
            "budget": round(budget, 2),
            "duration_days": duration,
            "start_month": int(start.month) if pd.notna(start) else np.nan,
            "start_weekday": int(start.weekday()) if pd.notna(start) else np.nan,
            "spend": round(spend, 2),
            "conversions": _conversions(metrics.get("actions"), objective),
        }
        row.update(_targeting_rows(adset.get("targeting")))
        rows.append(row)

    return pd.DataFrame(rows)
