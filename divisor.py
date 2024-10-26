import re
import os
import difflib

# Directory for output files
output_dir = "events_output"
os.makedirs(output_dir, exist_ok=True)

# Read the valid variables from stuff.txt
with open('stuff.txt', 'r') as file:
    valid_variables = set(re.findall(r'\b\w+\b', file.read()))

# Define allowed variables that should be ignored in the inconsistency check
allowed_variables = {
    "ai", "name", "title", "desc", "id", "show_in_missions", "mission_image",
    "image", "only_once", "highlight"
}

# Define trigger and outcome mappings
trigger_mapping = {
    "civ_economy_buildings_constructed_below": "economy_buildings_constructed_below",
    "civ_unrest_over": "civ_capital_unrest_over",
    "civ_province_unrest_over": "province_unrest_over",
    "civ_stability_over": "civ_capital_growth_rate_over",
    "civ_royal_marriages_over": "civ_allies_over",
    "province_population_all_below": "province_population_below",
    "province_economy_all_below": "province_economy_below",
    "civ_religion_over": "civ_capital_religion_is",
    "civ_administrative_buildings_constructed_over": "administrative_buildings_constructed_over",
    "civ_growth_rate_below": "civ_capital_growth_rate_below",
    "civ_infrastructure_over": "civ_capital_infrastructure_over",
    "civ_economy_buildings_constructed_over": "economy_buildings_constructed_over",
    "civ_stability": "civ_capital_growth_rate_over",
    "civ_research_over": "bonus_research",
    "make_vassal": "make_puppet",
    "remove_vassal": "is_not_puppet",
    "civ_ruler_personality": "civ_tag_government_is",
    "civ_advisors_recruited_over": "recruited_advisors_over"
}

outcome_mapping = {
    "bonus_general_defense": "bonus_generals_defense",
    "bonus_province_tax_efficiency_all": "province_tax_efficiency_all",
    "bonus_province_economy_capital": "province_economy_capital",
    "bonus_gold": "gold",
    "bonus_years_of_income": "gold_monthly_income",
    "bonus_capital_fort_level": "civ_capital_fort_level_over",
    "bonus_province_tax_efficiency_capital": "province_tax_efficiency_capital",
    "bonus_general_attack": "bonus_generals_attack",
    "bonus_province_unrest_all": "province_unrest_all",
    "bonus_province_growth_rate_all": "province_growth_rate_all",
    "bonus_technology_cost": "bonus_research",
    "bonus_province_religion_capital": "province_religion_capital",
    "bonus_prestige": "civ_prestige_over",
    "bonus_legacy": "legacy",
    "bonus_province_growth_rate_capital": "province_growth_rate_capital",
    "bonus_navy_tradition": "bonus_diplomacy_points",
    "bonus_stability": "bonus_growth_rate",
    "bonus_adm_power": "bonus_administration_buildings_cost",
    "bonus_province_population": "province_population",
    "bonus_military_advisor_skill": "civ_military_advisor_skill_over",
    "bonus_province_unrest_capital": "province_unrest_capital",
    "bonus_growth_rate_capital": "civ_capital_growth_rate_over",
    "bonus_province_defense_lvl_all": "province_defense_lvl_over",
    "bonus_province_devastation_capital": "province_devastation_capital",
    "bonus_infrastructure_capital": "civ_capital_infrastructure_over",
    "bonus_province_economy_all": "province_economy_all",
    "bonus_defense": "bonus_units_defense",
    "bonus_province_religion_all": "province_religion_all",
    "bonus_economy_buildings_constructed_over": "economy_buildings_constructed_over",
    "bonus_province_infrastructure_all": "province_infrastructure_all",
    "bonus_province_manpower_capital": "province_manpower_capital",
    "bonus_province_population_capital": "province_population_capital",
    "bonus_province_infrastructure_capital": "province_infrastructure_capital",
    "bonus_infrastructure_all": "province_infrastructure_all",
    "bonus_manpower": "civ_manpower_over",
    "bonus_province_economy_id": "province_economy_id"
}

# Read the input events from events.txt
with open('events.txt', 'r') as file:
    data = file.read()

# Split the data into individual events based on the 'id=' keyword
events = re.split(r'\nid=', data)
if events[0] == '':
    events.pop(0)  # Remove empty entry if the split adds one at the beginning

# Set to collect all structured variables used across all events
all_used_variables = set()
impacted_events_count = 0  # Counter for events with inconsistencies

# Pattern to match variables in the format: variable_name=value
structured_pattern = re.compile(r'\b(\w+)=[-\w\.]+')

# Iterate through each event, assign new ID, remove quotes, and save to a new file
for index, event in enumerate(events, start=1):
    # Define new ID
    new_id = f"MAK_{index:05d}"

    # Replace the old ID with the new one
    modified_event = re.sub(r'E_\d+', new_id, 'id=' + event, count=1)

    # Remove quotes from title, description, and buttons
    modified_event = re.sub(r'title="(.*?)"', r'title=\1', modified_event)
    modified_event = re.sub(r'desc="(.*?)"', r'desc=\1', modified_event)
    modified_event = re.sub(r'name="(.*?)"', r'name=\1', modified_event)

    # Find all variables matching the pattern
    structured_variables = set(structured_pattern.findall(modified_event))

    # Track modifications for inconsistent variables in this event
    event_modified = False
    for var in structured_variables:
        if var not in valid_variables and var not in allowed_variables:
            # Determine if the variable is a trigger or an outcome and apply the correct mapping
            if var in trigger_mapping:
                closest_match = trigger_mapping[var]
            elif var in outcome_mapping:
                closest_match = outcome_mapping[var]
            else:
                closest_match = None

            # Replace the inconsistent variable in the event text if a match was found
            if closest_match:
                modified_event = re.sub(r'\b' + re.escape(var) + r'\b', closest_match, modified_event)
                event_modified = True

    # If the event was modified, increase the impacted events count
    if event_modified:
        impacted_events_count += 1

    # Write each modified event to a new file
    with open(os.path.join(output_dir, f"{new_id}.txt"), 'w') as output_file:
        output_file.write(modified_event)

print("Event files created successfully with updated IDs, removed quotes, and corrected inconsistent variables.")

# Output the count of impacted events
print(f"Number of impacted events: {impacted_events_count}")
