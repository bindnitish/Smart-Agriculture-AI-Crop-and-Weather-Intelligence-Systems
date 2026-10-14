from agentic_ai.iot.devices import (
    automated_irrigation,
    soil_sensor,
    camera_drone,
    frost_alarm,
    lightning_detector,
    hail_protector,
    pest_detector,
    NDVI_sensor,
    weather_station
)
DEVICE_FUNCTIONS = {

    "automated_irrigation": automated_irrigation,

    "soil_sensor": soil_sensor,

    "camera_drone": camera_drone,

    "frost_alarm": frost_alarm,

    "lightning_detector": lightning_detector,

    "hail_protector": hail_protector,

    "pest_detector": pest_detector,

    "NDVI_sensor": NDVI_sensor,

    "weather_station": weather_station
}
from agentic_ai.rule_engine import (
    apply_combined_rule,
    apply_separate_rules
)


def make_decision(crop, weather):

    print("\n" + "=" * 60)
    print("AGENTIC AI DECISION")
    print("=" * 60)

    print(
        f"Predicted Crop    : {crop}"
    )

    print(
        f"Predicted Weather : {weather}"
    )


    # ======================================
    # 1. CHECK COMBINED RULE FIRST
    # ======================================

    combined_result = apply_combined_rule(
        crop,
        weather
    )


    # ======================================
    # 2. COMBINED RULE FOUND
    # ======================================

    if combined_result is not None:

        print(
            "\nDecision Type : COMBINED RULE"
        )

        print("\nIoT Devices:")

        for device in combined_result["devices"]:

            print(
                f" - {device}"
            )


        print("\nRecommended Actions:")

        for action in combined_result["actions"]:

            print(
                f" - {action}"
            )


        # ==================================
        # EXECUTE IoT ACTIONS
        # ==================================

        print("\nExecuting IoT Actions...")

        for device, action in zip(
            combined_result["devices"],
            combined_result["actions"]
        ):

            device_function = DEVICE_FUNCTIONS.get(
                device
            )

            if device_function is not None:

                device_function(action)

            else:

                print(
                    f"\nUnknown IoT device: {device}"
                )


        print("=" * 60)

        return combined_result


    # ======================================
    # 3. NO COMBINED RULE
    # ======================================

    print(
        "\nNo combined rule found."
    )

    print(
        "\nChecking separate rules..."
    )


    # ======================================
    # 4. APPLY SEPARATE RULES
    # ======================================

    separate_results = apply_separate_rules(
        crop,
        weather
    )


    print(
        "\nDecision Type : SEPARATE RULES"
    )


    # ======================================
    # 5. NO SEPARATE RULE
    # ======================================

    if not separate_results:

        print(
            "\nNo applicable rule found."
        )

        print("=" * 60)

        return None


    # ======================================
    # 6. DISPLAY SEPARATE RULES
    # ======================================

    for result in separate_results:

        print(
            f"\nRule Source : "
            f"{result['source']}"
        )

        print(
            f"Condition   : "
            f"{result['condition']}"
        )


        print("\nIoT Devices:")

        for device in result["devices"]:

            print(
                f" - {device}"
            )


        print("\nRecommended Actions:")

        for action in result["actions"]:

            print(
                f" - {action}"
            )


        # ==================================
        # EXECUTE SEPARATE IoT ACTIONS
        # ==================================

        print("\nExecuting IoT Actions...")

        for device, action in zip(
            result["devices"],
            result["actions"]
        ):

            device_function = DEVICE_FUNCTIONS.get(
                device
            )

            if device_function is not None:

                device_function(action)

            else:

                print(
                    f"\nUnknown IoT device: {device}"
                )


    print("=" * 60)

    return separate_results

    # ======================================
    # 1. COMBINED RULE FIRST
    # ======================================

    combined_result = apply_combined_rule(
        crop,
        weather
    )


    if combined_result is not None:

        print(
            "\nDecision Type : COMBINED RULE"
        )

        print("\nIoT Devices:")

        for device in combined_result["devices"]:

            print(
                f" - {device}"
            )

        print("\nRecommended Actions:")

        for action in combined_result["actions"]:

            print(
                f" - {action}"
            )
    # ======================================
    # EXECUTE IoT ACTIONS
    # ======================================

    print("\nExecuting IoT Actions...")

    for device, action in zip(
        combined_result["devices"],
        combined_result["actions"]
    ):

        device_function = DEVICE_FUNCTIONS.get(
            device
        )

        if device_function is not None:

            device_function(action)

        else:

            print(
                f"\nUnknown IoT device: {device}"
            )            

        print("=" * 60)

        return combined_result


    # ======================================
    # 2. FALLBACK TO SEPARATE RULES
    # ======================================

    separate_results = apply_separate_rules(
        crop,
        weather
    )


    print(
        "\nDecision Type : SEPARATE RULES"
    )


    if not separate_results:

        print(
            "\nNo applicable rule found."
        )

        print("=" * 60)

        return None


    for result in separate_results:

        print(
            f"\nRule Source : "
            f"{result['source']}"
        )

        print(
            f"Condition   : "
            f"{result['condition']}"
        )

        print("\nIoT Devices:")

        for device in result["devices"]:

            print(
                f" - {device}"
            )

        print("\nRecommended Actions:")

        for action in result["actions"]:

            print(
                f" - {action}"
            )


    print("=" * 60)

    return separate_results


    # ======================================
    # COMBINED RULE
    # ======================================

    if result is not None:

        print(
            "\nDecision Type     : COMBINED RULE"
        )

        print("\nIoT Devices:")

        for device in result["devices"]:

            print(
                f" - {device}"
            )

        print("\nRecommended Actions:")

        for action in result["actions"]:

            print(
                f" - {action}"
            )


    # ======================================
    # NO COMBINED RULE
    # ======================================

    else:

        print(
            "\nDecision Type     : NO COMBINED RULE"
        )

        print(
            "Separate rules will be checked."
        )


    print("=" * 60)


    return result