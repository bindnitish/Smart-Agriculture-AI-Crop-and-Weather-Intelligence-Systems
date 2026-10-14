# ==========================================
# SMART AGRICULTURE
# RULE-BASED AGENTIC AI
# ==========================================


# ==========================================
# COMBINED RULES
# Based on Paper Table 6
# ==========================================

COMBINED_RULES = {

    # --------------------------------------
    # Rice + Rain
    # --------------------------------------

    ("rice", "rain"): {
        "devices": [
            "automated_irrigation",
            "camera_drone"
        ],
        "actions": [
            "Postpone irrigation",
            "Monitor for waterlogging and disease"
        ]
    },


    # --------------------------------------
    # Wheat/Maize + Frost
    # --------------------------------------

    ("wheat", "frost"): {
        "devices": [
            "frost_alarm",
            "soil_sensor"
        ],
        "actions": [
            "Activate early frost warning for sensitive crops",
            "Delay sowing and check soil temperature"
        ]
    },

    ("maize", "frost"): {
        "devices": [
            "frost_alarm",
            "soil_sensor"
        ],
        "actions": [
            "Activate early frost warning for sensitive crops",
            "Delay sowing and check soil temperature"
        ]
    },


    # --------------------------------------
    # Sugarcane + Rainbow
    # --------------------------------------

    ("sugarcane", "rainbow"): {
        "devices": [
            "NDVI_sensor",
            "camera_drone"
        ],
        "actions": [
            "Assess crop health",
            "Visual inspection for fungus/insects"
        ]
    },


    # --------------------------------------
    # Maize + Hail
    # --------------------------------------

    ("maize", "hail"): {
        "devices": [
            "hail_protector",
            "camera_drone"
        ],
        "actions": [
            "Deploy hail covers or protection systems",
            "Inspect for physical crop damage"
        ]
    },


    # --------------------------------------
    # Wheat + Lightning
    # --------------------------------------

    ("wheat", "lightning"): {
        "devices": [
            "lightning_detector",
            "frost_alarm"
        ],
        "actions": [
            "Monitor for surge",
            "Delay field equipment use"
        ]
    },


    # --------------------------------------
    # Wheat/Sugarcane + Fogsmog
    # --------------------------------------

    ("wheat", "fogsmog"): {
        "devices": [
            "camera_drone",
            "pest_detector"
        ],
        "actions": [
            "Reduced photosynthesis alert",
            "Trap monitoring for insect growth"
        ]
    },

    ("sugarcane", "fogsmog"): {
        "devices": [
            "camera_drone",
            "pest_detector"
        ],
        "actions": [
            "Reduced photosynthesis alert",
            "Trap monitoring for insect growth"
        ]
    },


    # --------------------------------------
    # Jute/Rice + Sandstorm
    # --------------------------------------

    ("jute", "sandstorm"): {
        "devices": [
            "NDVI_sensor",
            "pest_detector"
        ],
        "actions": [
            "Assess plant stress",
            "Trap installation to protect from pest spread"
        ]
    },

    ("rice", "sandstorm"): {
        "devices": [
            "NDVI_sensor",
            "pest_detector"
        ],
        "actions": [
            "Assess plant stress",
            "Trap installation to protect from pest spread"
        ]
    },


    # --------------------------------------
    # Wheat/Jute + Dew
    # --------------------------------------

    ("wheat", "dew"): {
        "devices": [
            "camera_drone",
            "frost_alarm"
        ],
        "actions": [
            "Monitor for early fungal disease",
            "Send cold alerts"
        ]
    },

    ("jute", "dew"): {
        "devices": [
            "camera_drone",
            "frost_alarm"
        ],
        "actions": [
            "Monitor for early fungal disease",
            "Send cold alerts"
        ]
    },


    # --------------------------------------
    # Maize/Sugarcane + Glaze
    # --------------------------------------

    ("maize", "glaze"): {
        "devices": [
            "soil_sensor",
            "pest_detector"
        ],
        "actions": [
            "Delay sowing",
            "Monitor pest habitat shift"
        ]
    },

    ("sugarcane", "glaze"): {
        "devices": [
            "soil_sensor",
            "pest_detector"
        ],
        "actions": [
            "Delay sowing",
            "Monitor pest habitat shift"
        ]
    }
}


# ==========================================
# RULE ENGINE FUNCTION
# ==========================================

def apply_combined_rule(
    crop,
    weather
):

    crop = crop.lower()
    weather = weather.lower()

    key = (
        crop,
        weather
    )

    # --------------------------------------
    # Check combined rule
    # --------------------------------------

    if key in COMBINED_RULES:

        rule = COMBINED_RULES[key]

        return {
            "rule_type": "combined",
            "crop": crop,
            "weather": weather,
            "devices": rule["devices"],
            "actions": rule["actions"]
        }
    # ==========================================
# SEPARATE CROP RULES
# ==========================================

CROP_RULES = {

    "wheat": {
        "devices": [
            "pest_detector"
        ],
        "actions": [
            "Activate fungal spore detection"
        ]
    },

    "rice": {
        "devices": [
            "soil_sensor"
        ],
        "actions": [
            "Monitor soil water conditions"
        ]
    },

    "maize": {
        "devices": [
            "soil_sensor"
        ],
        "actions": [
            "Monitor crop field conditions"
        ]
    },

    "sugarcane": {
        "devices": [
            "NDVI_sensor"
        ],
        "actions": [
            "Assess vegetation health"
        ]
    },

    "jute": {
        "devices": [
            "camera_drone"
        ],
        "actions": [
            "Inspect crop condition"
        ]
    }
}


# ==========================================
# SEPARATE WEATHER RULES
# ==========================================

WEATHER_RULES = {

    "fogsmog": {
        "devices": [
            "camera_drone"
        ],
        "actions": [
            "Visual scan for early mildew or rust symptoms"
        ]
    },

    "frost": {
        "devices": [
            "frost_alarm"
        ],
        "actions": [
            "Activate frost warning"
        ]
    },

    "hail": {
        "devices": [
            "hail_protector"
        ],
        "actions": [
            "Deploy hail protection"
        ]
    },

    "lightning": {
        "devices": [
            "lightning_detector"
        ],
        "actions": [
            "Monitor for electrical surge"
        ]
    },

    "rain": {
        "devices": [
            "weather_station"
        ],
        "actions": [
            "Monitor rainfall conditions"
        ]
    },

    "sandstorm": {
        "devices": [
            "NDVI_sensor"
        ],
        "actions": [
            "Assess plant stress"
        ]
    },

    "dew": {
        "devices": [
            "camera_drone"
        ],
        "actions": [
            "Monitor for early fungal disease"
        ]
    },

    "glaze": {
        "devices": [
            "soil_sensor"
        ],
        "actions": [
            "Monitor soil conditions"
        ]
    },

    "rainbow": {
        "devices": [
            "camera_drone"
        ],
        "actions": [
            "Assess crop condition"
        ]
    },

    "rime": {
        "devices": [
            "frost_alarm"
        ],
        "actions": [
            "Monitor cold conditions"
        ]
    },

    "snow": {
        "devices": [
            "frost_alarm"
        ],
        "actions": [
            "Activate cold weather warning"
        ]
    }
}
# ==========================================
# APPLY SEPARATE RULES
# ==========================================

def apply_separate_rules(
    crop,
    weather
):

    crop = crop.lower()
    weather = weather.lower()

    results = []


    # ==========================================
    # Crop rule
    # ==========================================

    if crop in CROP_RULES:

        crop_rule = CROP_RULES[crop]

        results.append({
            "source": "crop",
            "condition": crop,
            "devices": crop_rule["devices"],
            "actions": crop_rule["actions"]
        })


    # ==========================================
    # Weather rule
    # ==========================================

    if weather in WEATHER_RULES:

        weather_rule = WEATHER_RULES[weather]

        results.append({
            "source": "weather",
            "condition": weather,
            "devices": weather_rule["devices"],
            "actions": weather_rule["actions"]
        })

    return results