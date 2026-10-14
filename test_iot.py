from agentic_ai.iot.devices import (
    automated_irrigation,
    soil_sensor,
    camera_drone
)


print("=" * 60)
print("SMART AGRICULTURE - IoT SIMULATION")
print("=" * 60)


automated_irrigation(
    "Postpone irrigation"
)

soil_sensor(
    "Check water saturation"
)

camera_drone(
    "Monitor crop for disease"
)


print("\n" + "=" * 60)
print("IoT SIMULATION COMPLETED")
print("=" * 60)