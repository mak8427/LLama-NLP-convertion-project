import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sentencepiece
import google.protobuf

# Load the Hugging Face token from token.txt
with open("token.txt", "r") as token_file:
    hf_token = token_file.read().strip()

# Set up LLaMA model and tokenizer using Hugging Face token
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA GPU is required to run this script.")

device = torch.device("cuda")
model_name = "llmware/bling-1b-0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def read_random_events(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def parse_events(lines):
    events = []
    current_event = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_event:
                events.append(current_event)
                current_event = []
        current_event.append(line.strip())
    if current_event:
        events.append(current_event)
    return events

def llama_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def convert_event_with_llama(event_lines):
    event_str = "\n".join(event_lines)


    print(event_str)
    effects = """
trigger_and
trigger_and_not
trigger_or
trigger_or_not
trigger_and_end
trigger_or_end
next_and
next_and_not
next_or
next_or_not
invested_in_economy_over=X
invested_in_economy_below=X
increased_growth_rate_over=X
increased_growth_rate_below=X
increased_tax_efficiency_over=X
increased_tax_efficiency_below=X
increased_manpower_over=X
increased_manpower_below=X
developed_infrastructure_over=X
developed_infrastructure_below=X
buildings_constructed_over=X
buildings_constructed_below=X
administrative_buildings_constructed_over=X
administrative_buildings_constructed_below=X
economy_buildings_constructed_over=X
economy_buildings_constructed_below=X
military_buildings_constructed_over=X
military_buildings_constructed_below=X
unique_capital_buildings_constructed_over=X
unique_capital_buildings_constructed_below=X
civ_conquered_provinces_over=X
civ_conquered_provinces_below=X
civ_wars_total_over=X
civ_wars_total_below=X
is_player=TAG
is_not_player=TAG
civ_religion_is=RELIGION_ID
civ_tag_religion_is=TAG=RELIGION_ID
civ_tag_religion_is_not=TAG=RELIGION_ID
civ_government_is=GOVERNMENT_ID
civ_tag_government_is=TAG=GOVERNMENT_ID
civ_tag_government_is_not=TAG=GOVERNMENT_ID
civ_provinces_over=X
civ_provinces_below=X
civ_provinces_equals=X
civ_population_over=X
civ_population_below=X
civ_economy_over=X
civ_economy_below=X
civ_gold_over=X
civ_gold_below=X
civ_allies_over=X
civ_allies_below=X
civ_defensive_pacts_over=X
civ_defensive_pacts_below=X
civ_non_aggression_pacts_over=X
civ_non_aggression_pacts_below=X
civ_vassals_over=X
civ_vassals_below=X
civ_neighbors_over=X
civ_neighbors_below=X
random_chance=PERCENT
is_civ=TAG
is_civ_m=TAG
civ_total_income_over=X
civ_total_income_below=X
civ_income_taxation_over=X
civ_income_taxation_below=X
civ_income_economy_over=X
civ_income_economy_below=X
civ_income_production_over=X
civ_income_production_below=X
civ_legacy_per_month_over=X
civ_legacy_per_month_below=X
civ_research_per_month_over=X
civ_research_per_month_below=X
civ_diplomacy_over=X
civ_diplomacy_below=X
civ_loans_over=X
civ_loans_below=X
civ_inflation_over=X
civ_inflation_below=X
civ_legacy_over=X
civ_legacy_below=X
civ_unlocked_legacies_over=X
civ_unlocked_legacies_below=X
civ_unlocked_technologies_over=X
civ_unlocked_technologies_below=X
civ_unlocked_advantages_over=X
civ_unlocked_advantages_below=X
civ_rank_position_over=X
civ_rank_position_below=X
civ_prestige_over=X
civ_prestige_below=X
civ_has_resource=RESOURCE_ID
civ_has_resource_over=RESOURCE_ID=X
civ_production_over=RESOURCE_ID=X
civ_largest_producer_over=RESOURCE_ID
largest_producer_production_over=RESOURCE_ID=X
civ_manpower_over=X
civ_manpower_below=X
civ_max_manpower_over=X
civ_max_manpower_below=X
civ_manpower_perc_over=PERCENT
civ_manpower_perc_below=PERCENT
civ_regiments_over=X
civ_regiments_below=X
civ_regiments_limit_over=X
civ_regiments_limit_below=X
civ_battle_width_over=X
civ_battle_width_below=X
civ_regiments_over_regiments_limit=true
civ_gold_over_max_amount_of_gold=true
civ_capital_buildings_over=X
civ_capital_buildings_below=X
civ_capital_tax_efficiency_over=X
civ_capital_tax_efficiency_below=X
civ_capital_economy_over=X
civ_capital_economy_below=X
civ_capital_manpower_over=X
civ_capital_manpower_below=X
civ_capital_growth_rate_over=X
civ_capital_growth_rate_below=X
civ_capital_infrastructure_over=X
civ_capital_infrastructure_below=X
civ_capital_population_over=X
civ_capital_population_below=X
civ_capital_fort_level_over=X
civ_capital_fort_level_below=X
civ_capital_income_over=X
civ_capital_income_below=X
civ_capital_unrest_over=X
civ_capital_unrest_below=X
civ_capital_religion_is=RELIGION_ID
civ_capital_has_building=BUILDING_ID=SUB_BUILDING_ID
civ_capital_is_occupied=true
civ_capital_is_not_occupied=true
civ_capital_is_under_siege=true
civ_capital_continent_is=CONTINENT_ID
civ_capital_continent_is_not=CONTINENT_ID
recruited_advisors_over=X
recruited_advisors_below=X
civ_administrative_advisor_skill_over=X
civ_administrative_advisor_skill_below=X
civ_economic_advisor_skill_over=X
civ_economic_advisor_skill_below=X
civ_innovation_advisor_skill_over=X
civ_innovation_advisor_skill_below=X
civ_military_advisor_skill_over=X
civ_military_advisor_skill_below=X
civ_advisor_age_over=ADVISOR_ID=X
civ_advisor_production_efficiency_over=ADVISOR_ID=X
civ_advisor_construction_cost_over=ADVISOR_ID=X
civ_military_academy_over=X
civ_military_academy_below=X
civ_military_academy_for_generals_over=X
civ_military_academy_for_generals_below=X
civ_capital_city_over=X
civ_capital_city_below=X
civ_supreme_court_over=X
civ_supreme_court_below=X
civ_nuclear_reactor_over=X
civ_nuclear_reactor_below=X
exists=TAG
exists_not=TAG
exists_any=TAG
exists_any_not=TAG
is_puppet=TAG
is_not_puppet=TAG
civ_is_vassal_of_civ=TAG1=TAG2
civ_is_at_war=true
civ_is_not_at_war=true
civ_is_at_war_days_over=DAYS
civ_has_more_provinces_than_civ=TAG1=TAG2
civ_has_larger_population_than_civ=TAG1=TAG2
civ_has_larger_economy_than_civ=TAG1=TAG2
civ_has_larger_regiments_limit_than_civ=TAG1=TAG2
civ_has_more_regiments_than_civ=TAG1=TAG2
civ_has_higher_ranking_than_civ=TAG1=TAG2
civ_has_more_technologies_than_civ=TAG1=TAG2
civs_opinion_over=TAG1=TAG2=VALUE
civs_opinion_below=TAG1=TAG2=VALUE
civs_are_neighbors=TAG1=TAG2
civs_are_not_neighbors=TAG1=TAG2
civs_are_rivals=TAG1=TAG2
civ_has_rivalry=TAG1=TAG2
civ_has_rivalry_not=TAG1=TAG2
civs_are_at_war=TAG1=TAG2
civs_are_not_at_war=TAG1=TAG2
civs_have_defensive_pact=TAG1=TAG2
civs_have_non_aggression=TAG1=TAG2
civs_have_alliance=TAG1=TAG2
civs_have_truce=TAG1=TAG2
civ_have_military_access=TAG1=TAG2
civ_have_guarantee=TAG1=TAG2
playing_time_over=DAYS
playing_time_below=DAYS
exact_day=DAY=MONTH=YEAR
year_over=YEAR
year_below=YEAR
province_controlled_by=PROVINCE_ID=TAG
province_not_controlled_by=PROVINCE_ID=TAG
province_civ_has_core=PROVINCE_ID=TAG
province_economy_over=PROVINCE_ID=VALUE
province_economy_below=PROVINCE_ID=VALUE
province_growth_rate_over=PROVINCE_ID=VALUE
province_growth_rate_below=PROVINCE_ID=VALUE
province_population_over=PROVINCE_ID=VALUE
province_population_below=PROVINCE_ID=VALUE
province_tax_efficiency_over=PROVINCE_ID=VALUE
province_tax_efficiency_below=PROVINCE_ID=VALUE
province_manpower_over=PROVINCE_ID=VALUE
province_manpower_below=PROVINCE_ID=VALUE
province_income_over=PROVINCE_ID=VALUE
province_income_below=PROVINCE_ID=VALUE
province_religion_is=PROVINCE_ID=RELIGION_ID
province_religion_is_not=PROVINCE_ID=RELIGION_ID
province_unrest_over=PROVINCE_ID=VALUE
province_unrest_below=PROVINCE_ID=VALUE
province_infrastructure_over=PROVINCE_ID=VALUE
province_infrastructure_below=PROVINCE_ID=VALUE
province_is_occupied=PROVINCE_ID
province_is_not_occupied=PROVINCE_ID
province_is_under_siege=PROVINCE_ID
province_buildings_over=PROVINCE_ID=VALUE
province_buildings_below=PROVINCE_ID=VALUE
province_buildings_limit_over=PROVINCE_ID=VALUE
province_buildings_limit_below=PROVINCE_ID=VALUE
province_defense_lvl_over=PROVINCE_ID=VALUE
province_defense_lvl_below=PROVINCE_ID=VALUE
province_is_capital=PROVINCE_ID
province_has_building=PROVINCE_ID=BUILDING_ID=SUB_BUILDING_ID
"""
    outcomes = """
play_music=KOL_DC_01
gold=125
gold_monthly_income=2.0
legacy=75
legacy_monthly=4.5
player_set_civ=fra
research=35
advantage_points=3
manpower=7500
inflation=8.2
ai_aggression=100
ai_aggression=-1000
ae_set=24.0
move_capital=75
declare_war=fra
declare_war2=ger=pol
add_alliance=ger=hun
add_non_aggression=fra=spa
add_military_access=rus=swe
add_guarantee=fra=pol
add_defensive_pact=ita=ger
add_truce=ita=ger
change_ideology=4
change_ideology_civ=2=fra
change_religion=3
change_religion_civ=3=fra
annex=0;1;2;3;4;
annex_from_civ=fra=0;1;2;3;4;
annex_by_civ_from_civ=ger=czsl=646;652;647;
annex_civ=fra
annexed_by_civ=ger
make_puppet=ger=atr
set_civ_tag=pol
set_civ_tag=spa_n
set_civ_tag=fra_r
set_civ_tag2=czsl=slo_z
set_civ_tag2=fra=fra_f
run_event=the_austrian_question
add_general=true
add_general2=Zhukov
add_general3=Galileo=6=4
add_general3=Galileo=-1=3
add_advisor2=1=Franklin
add_advisor2=3=Zhukov
add_advisor=1
add_ruler=Lukasz=Jakowski=223=23=06=1992
add_ruler=TheName=TheSurname=Zedong=23=06=1992
province_economy=15.4
province_economy_capital=27.2
province_economy_all=3.5
province_economy_id=11245=4.8
province_tax_efficiency=3.2
province_tax_efficiency_capital=4.7
province_tax_efficiency_all=2.5
province_tax_efficiency_id=725=3.0
province_manpower=1.2
province_manpower_capital=1.7
province_manpower_all=3.5
province_manpower_id=332=4.0
province_growth_rate=8.2
province_growth_rate_capital=6.2
province_growth_rate_all=7.25
province_growth_rate_id=725=3.0
province_population=5000
province_population_capital=12500
province_population_all=2500
province_religion=3
province_religion_capital=2
province_religion_all=3
province_religion_id=1000=4
province_devastation=27.5
province_devastation_capital=45.5
province_devastation_all=15
province_devastation_id=35=25.0
province_unrest=15.5
province_unrest_capital=35.5
province_unrest_all=25.0
province_unrest_id=2541=-25.0
province_infrastructure=2
province_infrastructure_capital=1
province_infrastructure_all=1
province_infrastructure_id=332=1
province_add_core_civ=332=pol
province_remove_core_civ=332=pol
add_new_army=0=1=3=2
add_variable=capital_city_economic_development
add_variable2=ger=sudetenland_accepted
bonus_duration=2
bonus_monthly_income=24.4
bonus_monthly_income=12.2
bonus_monthly_legacy=2.4
bonus_monthly_legacy_percentage=10.2
bonus_maximum_amount_of_gold=500
bonus_tax_efficiency=10.5
bonus_growth_rate=15
bonus_province_maintenance=-25
bonus_buildings_maintenance_cost=-10
bonus_maintenance_cost=2.2
bonus_production_efficiency=25
bonus_income_production=10
bonus_income_taxation=7.5
bonus_income_economy=10.5
bonus_corruption=10.0
bonus_inflation=5.0
bonus_research=10
bonus_research_points=4.5
bonus_max_manpower=10000
bonus_max_manpower_percentage=10.25
bonus_manpower_recovery_speed=25
bonus_reinforcement_speed=15
bonus_army_morale_recovery=25
bonus_army_maintenance=-25
bonus_recruitment_time=-15
bonus_recruit_army_cost=-20
bonus_recruit_army_first_line_cost=-15
bonus_recruit_army_second_line_cost=-25
bonus_generals_attack=2
bonus_generals_defense=1
bonus_units_attack=4
bonus_units_defense=3
bonus_regiments_limit=6
bonus_battle_width=4
bonus_discipline=7.5
bonus_siege_effectiveness=25.5
bonus_max_morale=20
bonus_army_movement_speed=30.4
bonus_manpower_recovery_from_disbanded_army=25
bonus_war_score_cost=-25
bonus_construction_cost=-10
bonus_administration_buildings_cost=-15
bonus_military_buildings_cost=-20
bonus_economy_buildings_cost=-25
bonus_construction_time=-25
bonus_invest_in_economy_cost=-13.75
bonus_increase_manpower_cost=-7.25
bonus_increase_tax_efficiency_cost=-15.4
bonus_develop_infrastructure_cost=-12.5
bonus_increase_growth_rate_cost=-7.25
bonus_diplomacy_points=25
bonus_improve_relations_modifier=15
bonus_income_from_vassals=25
bonus_aggressive_expansion=-25
bonus_core_cost=-25
bonus_religion_cost=-15
bonus_revolutionary_risk=25
bonus_loan_interest=-25
bonus_loans_limit=1
bonus_all_characters_life_expectancy=5
bonus_advisor_cost=-20
bonus_advisors_max_level=2
bonus_general_cost=-15
bonus_disease_death_rate=75
join_alliance_special_id_first_tier=0
join_alliance_special_id_second_tier=1
leave_alliance_special_id=0
kill_ruler=true
kill_ruler_chance=35
kill_advisor=1
promote_advisor=2
military_academy=1
military_academy=-2
military_academy_generals=3
capital_city_level=1
supreme_court=2
nuclear_reactor=1
explode=ming
price_change_random=15=16=12=36
price_change_random=5=35=24=36
price_change_random_up=6=24=21=0
price_change_random_down=40=25=12=4
price_change=0=10=15=12=24
price_change=9=20=12=96=12
price_change_up=0=20=40=18=24
price_change_down=0=20=40=18=24
price_change_group=0=15=5=24=12
price_change_group_up=3=15=5=24=12
price_change_group_down=4=15=5=24=12
"""
    example= """id=J_RH
title=rich_harvest.t
desc=rich_harvest.d

image=76.png

show_in_missions=true
mission_image=4

only_once=true

trigger_and
next_and
random_chance=13.5
civ_production_over=0=29
trigger_and_end

option_btn
name=Excellent
ai=25
province_growth_rate=2.8
bonus_duration=5
bonus_growth_rate=4
option_end

option_btn
name=Amazing
ai=25
province_growth_rate=3.2
bonus_duration=5
bonus_growth_rate=2
bonus_increase_growth_rate_cost=-5.0
option_end

id=J_abre
title=advisors_research_breakthrough.t
desc=advisors_research_breakthrough.d
mission_desc=advisors_research_breakthrough.m

image=44.png

show_in_missions=true
mission_image=3

only_once=true

trigger_and
next_and
civ_innovation_advisor_skill_over=1
next_and
random_chance=27.5
trigger_and_end

option_btn
name=ResearchAcceleration
ai=10
bonus_duration=10
bonus_research_points=5.0
option_end

option_btn
name=StrengthenOurLegacy
ai=50
bonus_duration=10
bonus_monthly_legacy=0.6
option_end

"""


    prompt = f" Make another similar event using those possible triggers and effect: {effects} {outcomes} WITH THIS EXAMPLE:{example} Create a new  economic event:  id=q_04"
    generated_event = llama_generate(prompt)
    return generated_event

def process_and_save_events(input_file, output_folder):
    lines = read_random_events(input_file)
    events = parse_events(lines)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, event in enumerate(events):
        adapted_event = convert_event_with_llama(event)
        event_file_path = os.path.join(output_folder, f"event_{idx + 1}.txt")
        with open(event_file_path, 'w') as file:
            file.write(adapted_event)

# File paths
input_file = 'RandomEvents.txt'
output_folder = 'AdaptedEvents'

# Process events and save
process_and_save_events(input_file, output_folder)
