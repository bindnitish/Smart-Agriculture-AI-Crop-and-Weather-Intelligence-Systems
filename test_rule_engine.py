from agentic_ai.decision import make_decision


print("=" * 60)
print("TESTING FALLBACK / SEPARATE RULE")
print("=" * 60)


# ==========================================
# TEST CASE
# ==========================================

crop = "unknown_crop"
weather = "unknown_weather"


result = make_decision(
    crop,
    weather
)