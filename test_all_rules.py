from agentic_ai.rule_engine import apply_combined_rule


# ==========================================
# TESTING ALL COMBINED RULES
# ==========================================

TEST_CASES = [

    ("rice", "rain"),

    ("wheat", "frost"),

    ("maize", "frost"),

    ("sugarcane", "rainbow"),

    ("maize", "hail"),

    ("wheat", "lightning"),

    ("wheat", "fogsmog"),

    ("sugarcane", "fogsmog"),

    ("jute", "sandstorm"),

    ("rice", "sandstorm"),

    ("wheat", "dew"),

    ("jute", "dew"),

    ("maize", "glaze"),

    ("sugarcane", "glaze"),
]


print("=" * 60)
print("TESTING ALL COMBINED RULES")
print("=" * 60)


passed = 0
failed = 0


for crop, weather in TEST_CASES:

    print("\n" + "-" * 60)

    print(
        f"Crop    : {crop}"
    )

    print(
        f"Weather : {weather}"
    )


    result = apply_combined_rule(
        crop,
        weather
    )


    if result is not None:

        passed += 1

        print(
            "Status  : PASS"
        )

        print(
            "Devices :",
            result["devices"]
        )

        print(
            "Actions :",
            result["actions"]
        )

    else:

        failed += 1

        print(
            "Status  : FAIL"
        )

        print(
            "No combined rule found."
        )


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("RULE TEST SUMMARY")
print("=" * 60)

print(
    f"Total Rules Tested : {len(TEST_CASES)}"
)

print(
    f"Passed             : {passed}"
)

print(
    f"Failed             : {failed}"
)

print("=" * 60)