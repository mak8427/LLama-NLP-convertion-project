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
model_name = "meta-llama/Llama-3.2-1B"

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
    output = model.generate(**inputs, max_new_tokens=400)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def convert_event_with_llama(event_lines):
    event_str = "\n".join(event_lines)


    print(event_str)
    test= '''######################################### Read: mods_steam_workshop.txt !#### To create a new mod for submission to the Steam Workshop, first create a new folder for your mod within the 'mods' directory.## Then, copy or recreate all the modified files into that folder!#### For example, if a file was originally located in game/ExampleFile.json,## it should be placed in mods/YourModName/game/ExampleFile.json in your mod's folder.#######For your mod the path will be:mods/YOUR_MOD_NAME/game/events/common/#####How to create new Event/Mission##### STEP 1 #####Create new txt file in path:game/events/common/Or if is it event for scenario the path will be:map/Earth3/scenarios/WW2/events/common/Where Earth3 is the map, and WW2 is the scenario.Exmaple file name: economys_golden_milestone.txtIt will be:game/events/common/economys_golden_milestone.txt->mods/NAME_OF_YOUR_MOD/game/events/common/economys_golden_milestone.txtormap/Earth3/scenarios/WW2/events/common/economys_golden_milestone.txt->mods/NAME_OF_YOUR_MOD/map/Earth3/scenarios/WW2/events/common/economys_golden_milestone.txt##### STEP 2 #####DO NOT USE SPACES IN CODE!!// -- EXAMPLE CODEid=economys_golden_milestonetitle=economys_golden_milestone.tdesc=economys_golden_milestone.dmission_desc=economys_golden_milestone.mpopUp=false// -- END CODEid=NAME_OF_THE_FILEtitle=NAME_OF_THE_FILE.tdesc=NAME_OF_THE_FILE.dmission_desc=NAME_OF_THE_FILE.mpopUp=false#Importantmission_desc, IS OPTIONAL, ONLY REQUIRED IF THIS IS A MISSION TO SHOW IN MISSION VIEWWe have now added an event id (It is only for the game), title(key), description(key) and optional mission description(key)In this example I will be using "economys_golden_milestone" as NAME_OF_THE_FILEpopUp=truepopUp=false# If true, yhe event will pop up on the screen! Default is false.possible_to_run=false# If false, then the event won't run at all, can be run only by other eventin event option use:run_event=german_ultimatumwhere "german_ultimatum" is id of the event to run.####################### STEP 3 ####### STEP 3 v1 -> Simple, text won't be translatabletitle=Economy's Golden Milestonedesc=Revitalize your Civilization's economy by strategic investments to reach a new era of prosperity.mission_desc=By strategically investing in key sectors, you have successfully revitalized your Civilization's economy.#### END STEP 3 v2###### SECOND WAY TO SET EVENT TITLE, DESCRIPTION AND OPTIONAL MISSION DESCRIPTION## STEP 3 v2 -> Use if: the text needs to be translatable, the code will look like this:title=economys_golden_milestone.tdesc=economys_golden_milestone.dmission_desc=economys_golden_milestone.mAdd a translation of the created keys for the title, description, and optional mission descriptionEdit file: game/languages/Bundle.propertiesAt the end of the file add newly created keys.economys_golden_milestone.t = Economy's Golden Milestoneeconomys_golden_milestone.m = Revitalize your Civilization's economy by strategic investments to reach a new era of prosperity.economys_golden_milestone.d = By strategically investing in key sectors, you have successfully revitalized your Civilization's economy.#### END STEP 3 v2####################### STEP 4 #####// -- EXAMPLE CODEimage=economy.png// -- END CODEIn this step we can select an event image.image=IMAGE_FILEYou can add own image or use existing image.Folders with event images:game/events/images/H/game/events/images/XH/game/events/images/XXH/##### STEP 5 #####// -- EXAMPLE CODEshow_in_missions=truemission_image=0only_once=true// -- END CODE# Show the event in missions view in the game.show_in_missions=trueshow_in_missions=false# The image on the button in the mission view of this event. Images folder: game/events/imagesMissions/H/mission_image=2mission_image=4etc.# The event will only be activated once or multiple times through the game.only_once=trueonly_once=false##### STEP 6 #####// -- EXAMPLE CODEtrigger_andnext_andinvested_in_economy_over=1next_orciv_provinces_over=10trigger_and_endtrigger_andnext_andciv_gold_over=75trigger_and_end// -- END CODEIn this example we have two separate triggers.The first trigger will be activated if the Civilization invests more than 1 in the economy or Civilization have more than 10 provinces. THIS TRIGGER IS AND(trigger_and)The second trigger will be activated if the Civilization has more than 75 gold in its budget. THIS TRIGGER IS AND(trigger_and)In this case the event will pop up if both triggers will be true.Let's create a trigger for the event# START OF THE TRIGGER 'AND'trigger_and# INSIDE TRIGGER CODEtrigger_and_end# END OF THE TRIGGER 'AND'# You can use multiple triggers in one event## You can choose from:trigger_andtrigger_and_endtrigger_and_nottrigger_and_not_endtrigger_ortrigger_or_endtrigger_or_nottrigger_or_not_end### INSIDE TRIGGER CODE# START OF THE INSIDE TRIGGER 'AND'next_andinvested_in_economy_over=1# END OF THE INSIDE TRIGGER 'AND'next_and# The next line inside the trigger will be "and"invested_in_economy_over=1# In this example, A civilization must invest more than 1 economy point in its economy.You must always choose what type the next inside trigger will be, in this case we have used "next_and".## You can choose from:next_andnext_and_notnext_ornext_or_not# You can use multiple inside triggers in one trigger########################### You can choose from, list with examples:has_variable=some_variablehas_variable_not=some_variable# Every time a Civilization runs an event (makes a decision), the 'id' of the event is stored in the Civilization's database of variables.# Example of an 'id' from the event file:# id=unique_id_of_event## This allows you to use:has_variable=unique_id_of_event# where 'unique_id_of_event' refers to the 'id' from that event file.#### Civinvested_in_economy_over=125invested_in_economy_below=50increased_growth_rate_over=25increased_growth_rate_below=75increased_tax_efficiency_over=50increased_tax_efficiency_below=100increased_manpower_over=150increased_manpower_below=250developed_infrastructure_over=10developed_infrastructure_below=55buildings_constructed_over=20buildings_constructed_below=20administrative_buildings_constructed_over=5administrative_buildings_constructed_below=15economy_buildings_constructed_over=1economy_buildings_constructed_below=8military_buildings_constructed_over=3military_buildings_constructed_below=5unique_capital_buildings_constructed_over=1unique_capital_buildings_constructed_below=3civ_conquered_provinces_over=10civ_conquered_provinces_below=16civ_wars_total_over=4civ_wars_total_below=10is_player=ming# true if player is playing as Mingis_not_player=fra# true if player is not playing as Franceciv_religion_is=3# true if the Civ has the religion ID: 3 (Protestant), as defined in game/Religions.json, with IDs starting from 0civ_tag_religion_is=bel=3# true if the Civ with the tag 'bel'(Belgium) has the religion ID: 3 (Protestant), as defined in game/Religions.json, with IDs starting from 0civ_tag_religion_is_not=fra=0# fra is France, 0 is Paganciv_government_is=0# true if Civ's government is ID: 0 -> Democracy, as defined in game/Governments.json, with IDs starting from 0civ_tag_government_is=usa=3# true if the Civ with the tag 'usa' has the government ID: 3 (Communism), as defined in game/Governments.json, with IDs starting from 0civ_tag_government_is_not=vie=3# true if the Civ with the tag 'vie' (Vietnam) does not have the government ID: 3 (Communism), as defined in game/Governments.json, with IDs starting from 0# You can create a mission to stop or start the spread of a government type by using these triggersciv_provinces_over=150civ_provinces_below=75civ_provinces_equals=35civ_population_over=1999641civ_population_below=1999641# Civ's total population is below 1999641civ_economy_over=1000civ_economy_below=79# Civ's total economy is below 79civ_gold_over=1234civ_gold_below=4321# Civ Diplomacyciv_allies_over=0civ_allies_below=1civ_defensive_pacts_over=1civ_defensive_pacts_below=1civ_non_aggression_pacts_over=1civ_non_aggression_pacts_below=1civ_vassals_over=0civ_vassals_below=3civ_neighbors_over=4civ_neighbors_below=2random_chance=12.5# 12.5% chance to be trueis_civ=spa# Civilization's TAG is spa -> Spainis_civ=spa_m# Civilization's TAG is spa_mciv_total_income_over=11.4civ_total_income_below=3.25civ_income_taxation_over=12.7civ_income_taxation_below=7.25civ_income_economy_over=13.23civ_income_economy_below=10.14civ_income_production_over=25.5civ_income_production_below=17.23civ_legacy_per_month_over=2.4civ_legacy_per_month_below=1.0civ_research_per_month_over=8.2civ_research_per_month_below=3.4civ_diplomacy_over=85.6civ_diplomacy_below=125.0civ_loans_over=0civ_loans_below=2civ_inflation_over=3.75civ_inflation_below=1.0civ_legacy_over=250civ_legacy_below=100civ_unlocked_legacies_over=4civ_unlocked_legacies_below=2civ_unlocked_technologies_over=23civ_unlocked_technologies_below=64civ_unlocked_advantages_over=4civ_unlocked_advantages_below=1civ_rank_position_over=14civ_rank_position_below=100civ_prestige_over=500civ_prestige_below=98civ_has_resource=3civ_has_resource_over=5=14.5# Civilization has a resource ID: 5 and its production is over 14.5civ_production_over=0=50# Civilization has a resource ID: 0 and its production is over 50, it is the same as civ_has_resource_over!..civ_largest_producer_over=6# Civilization is largest producer of over 6 Resourceslargest_producer_production_over=10=74# Largest producer of Resource ID: 10(Tea), produces more than 74## ResourceID can be find in: game/resources/Resources.json## ID: 10 will be Tealargest_producer_production_over=0=524# Largest producer of Resource ID: 0(Grain), produces more than 524civ_is_largest_producer=0# Civilization is largest producer of ResourceID: 0(Grain)civ_manpower_over=15000civ_manpower_below=1500civ_max_manpower_over=20000civ_max_manpower_below=17500civ_manpower_perc_over=50.2# Civilization's Manpower / Max manpower > 50.2%civ_manpower_perc_below=24.4# Civilization's Manpower / Max manpower < 24.4%civ_regiments_over=40civ_regiments_below=10civ_regiments_limit_over=24civ_regiments_limit_below=30civ_battle_width_over=20civ_battle_width_below=30civ_regiments_over_regiments_limit=trueciv_gold_over_max_amount_of_gold=true# Civ's capital provinceciv_capital_buildings_over=3civ_capital_buildings_below=4civ_capital_tax_efficiency_over=17.2civ_capital_tax_efficiency_below=24.3civ_capital_economy_over=24.2civ_capital_economy_below=92.2civ_capital_manpower_over=17.2civ_capital_manpower_below=21.4civ_capital_growth_rate_over=72.5civ_capital_growth_rate_below=97.2civ_capital_infrastructure_over=2civ_capital_infrastructure_below=3civ_capital_population_over=95000civ_capital_population_below=35750civ_capital_fort_level_over=3civ_capital_fort_level_below=5civ_capital_income_over=12.1civ_capital_income_below=3.2civ_capital_unrest_over=12.1civ_capital_unrest_below=1.0civ_capital_religion_is=0# true if the civilization's capital province religion is 0 (Pagan)civ_capital_has_building=8=1# true if Civ's capital province has constructed building: 8, 1# From the file: game/buildings/Buildings.json# Building number 8 (counted from 0), and within that building, ID 1# This corresponds to an Amphitheaterciv_capital_is_occupied=true# Civ's capital province is occupiedciv_capital_is_not_occupied=true# Civ's capital province is NOT occupiedciv_capital_is_under_siege=true# Civ's capital province is under siegeciv_capital_continent_is=3# true if The Civ's capital province is located on continent ID: 3 (North America), as defined in map/THE_MAP/Continents.json, with IDs counted from 0civ_capital_continent_is_not=3recruited_advisors_over=27recruited_advisors_below=23civ_administrative_advisor_skill_over=1civ_administrative_advisor_skill_below=2civ_economic_advisor_skill_over=3civ_economic_advisor_skill_below=4civ_innovation_advisor_skill_over=2civ_innovation_advisor_skill_below=5civ_military_advisor_skill_over=3civ_military_advisor_skill_below=3civ_advisor_age_over=2=37# Advisor ID: 2, Years old is over 37# 0 = Administrative Advisor# 1 = Economic Advisor# 2 = Innovation Advisor# 3 = Military Advisorciv_advisor_production_efficiency_over=1=4.5# Advisor ID: 1, Production Efficiency old is over 4.5civ_advisor_construction_cost_over=1=-8civ_military_academy_over=0civ_military_academy_below=3civ_military_academy_for_generals_over=1civ_military_academy_for_generals_below=2civ_capital_city_over=1civ_capital_city_below=3civ_supreme_court_over=2civ_supreme_court_below=1civ_nuclear_reactor_over=2civ_nuclear_reactor_below=3exists=pol_m## Exists, has more than 0 provinces, and the tag is exactly "pol_m"exists_not=pol_m## The Civ with the tag 'pol_m' has 0 provincesexists_any=eng_m## Exists, has more than 0 provinces, and the tag is "eng_m" or any "eng"exists_any_not=eng## The Civ with the tag 'eng_m' or any 'eng' has 0 provinces## true if Civ with TAG "fra" is a Vassal/Puppetis_puppet=fra## true if Civ with TAG "fra" is not a Vassal/Puppetis_not_puppet=fraciv_is_vassal_of_civ=fra=ger## true if Civ with TAG "fra" is a Vassal/Puppet of civ with TAG "ger"civ_is_at_war=true# true if civ is at warciv_is_not_at_war=true# true if civ is not at warciv_is_at_war_days_over=25civ_has_more_provinces_than_civ=pol=fraciv_has_larger_population_than_civ=fra=gerciv_has_larger_economy_than_civ=pol_m=fraciv_has_larger_regiments_limit_than_civ=ger=fraciv_has_more_regiments_than_civ=pol=rusciv_has_higher_ranking_than_civ=rus=gerciv_has_more_technologies_than_civ=rus=fracivs_opinion_over=ger=pol=-25.0# true if 'ger' (Germany) and 'pol' (Poland) have an opinion over -25.0civs_opinion_below=ger=pol=-25.0# true if 'ger' (Germany) and 'pol' (Poland) have an opinion below -25.0civs_are_neighbors=ger=fra# true if 'ger' (Germany) and 'fra' (France) share a border.civs_are_not_neighbors=ger=fracivs_are_rivals=ger=pol# true if 'ger' (Germany) and 'pol' (Poland) have each other set as rivals.civ_has_rivalry=ger=pol# true if 'ger' (Germany) has 'pol' (Poland) set as a rival.civ_has_rivalry_not=ger=polcivs_are_at_war=ger=fra# true if 'ger'(Germany) and 'fra'(France) are at warcivs_are_not_at_war=ger=fra# true if 'ger'(Germany) and 'fra'(France) are not at warcivs_have_defensive_pact=ger=fra# true if 'ger'(Germany) and 'fra'(France) have a Defensive Pactcivs_have_non_aggression=ger=fra# true if 'ger'(Germany) and 'fra'(France) have a Non Aggression Pactcivs_have_alliance=ger=fra# true if 'ger'(Germany) and 'fra'(France) have an Alliancecivs_have_truce=ger=fra# true if 'ger'(Germany) and 'fra'(France) have Truceciv_have_military_access=ger=fra# true if 'ger'(Germany) have Military Access to 'fra'(France)civ_have_guarantee=ger=fra# true if 'ger'(Germany) Guarantee Independence of 'fra'(France)playing_time_over=365playing_time_below=365## The day in the game is: 	1 September 1939exact_day=1=9=1939## The current in-game year is > 1800year_over=1800## The current in-game year is < 1750year_below=1750# A  civilization with the TAG 'ger' (Germany) has this variable; it can be used to create a chain of events.has_variable_civ=ger=the_sudetenland_question## Provinceprovince_controlled_by=1024=fra# true if Province ID: 1024 is controlled by the civilization with civTag: 'fra'province_not_controlled_by=1024=ger# true if Province ID: 1024 is NOT controlled by the civilization with civTag: 'ger'province_civ_has_core=728=pol# true if the civilization with civTag: 'pol' has a core in Province ID: 728province_economy_over=464=12.1# true if Province ID: 464 has economy over 12.1province_economy_below=464=3.2# true if Province ID: 464 has economy below 3.2province_growth_rate_over=321=60.0# true if Province ID: 321 has Growth Rate over 60.0province_growth_rate_below=224=12.5province_population_over=1244=90123# true if Province ID: 1244 has Population over 90123province_population_below=1244=72321province_tax_efficiency_over=7227=30.0# true if Province ID: 7227 has Tax efficiency over 30.0province_tax_efficiency_below=624=2.6province_manpower_over=632=6.4# true if Province ID: 632 has Manpower Level over 6.4province_manpower_below=621=1.2province_income_over=38=12.4# true if Province ID: 38 has income from province over 12.4province_income_below=1664=3.4province_religion_is=1322=0# true if Province ID: 1332 has religion ID: 0(Pagan)province_religion_is_not=7224=1# true if Province ID: 7224 has not religion ID: 1(Catholic)province_unrest_over=776=60.5# true if the unrest in Province ID: 776 is over 60.5province_unrest_below=124=8.5# true if the unrest in Province ID: 124 is below 8.5province_infrastructure_over=2556=1# true if the infrastructure in Province ID: 2556 is over 1province_infrastructure_below=333=1# true if the infrastructure in Province ID: 333 is below 1province_is_occupied=8224# true if the Province ID: 8224 is occupiedprovince_is_not_occupied=8224# true if the Province ID: 8224 is not occupiedprovince_is_under_siege=724# true if the Province ID: 8224 is under siegeprovince_buildings_over=777=3# true if the Province ID: 777 has more than 3 constructed buildingsprovince_buildings_below=776=2# true if the Province ID: 776 has less than 2 constructed buildingsprovince_buildings_limit_over=2213=8# true if Province ID: 2213 has a building limit of over 8province_buildings_limit_below=2100=6# true if Province ID: 2100 has a building limit of below 6province_defense_lvl_over=8224=0# true if the defense level in Province ID: 8224 is over 0province_defense_lvl_below=327=1# true if the defense level in Province ID: 327 is below 1province_is_capital=624# true if Province ID: 624 is the capital provinceprovince_has_building=464=8=1# true if Province ID: 464 has constructed building: 8, 1# From the file: game/buildings/Buildings.json# Building number 8 (counted from 0), and within that building, ID 1# This corresponds to an Amphitheater##########join_alliance_special_id_first_tier=0# Join Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).join_alliance_special_id_second_tier=1# Join Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).leave_alliance_special_id=0# Leave Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).##################################### STEP 7 #####// -- EXAMPLE CODEoption_btnname=GoodNewsai=20gold=250legacy=25add_variable=capital_city_economic_developmentbonus_duration=5bonus_monthly_income=12.4option_endoption_btnname=EconomicTriumphai=10province_economy_capital=10legacy=50bonus_duration=2bonus_monthly_income=24.4option_end// -- END CODENow let's add event options (buttons) for the Civilization to select.In the example code we have two buttons. We will work on the first button.# START OF THE BUTTONoption_btn# INSIDE BUTTON CODEoption_end# END OF THE BUTTON### Let's go through the "INSIDE BUTTON CODE" section.## To set the text that will be displayed on the button:name=YOUR_TEXT_KEY# Button text translations are added in the same file as the event title, step 3.# Example:name=GoodNewsSo in the translaton file we will add:GoodNews = Good News## We can then add what the probability is that the AI will choose this option:ai=20## And now we add the event results of this option# Example:province_economy_capital=10# Civilization will get 10 economy points in the capital province.legacy=50# By choosing this option, the Civilization will receive 50 legacy points.# etc.##################################### List of all available, with simple examples:play_music=KOL_DC_01# Play music file from audio/music/# or mods/YourMod/audio/music/# Don't put the extension of the file. The file format must be .ogggold=125gold_monthly_income=2.0legacy=75# Civ will receive 75 legacy pointslegacy_monthly=4.5# Civ will get 4.5 monthly legacyplayer_set_civ=fra# Player will now play as Franceresearch=35# Civ will get 35 research pointsadvantage_points=3# Civ will get 3 advantage pointsmanpower=7500# +7500 manpowerinflation=8.2# +8.2 inflationai_aggression=100# AI Aggression: +100 out of 1000ai_aggression=-1000# AI Aggression: -1000 out of 1000ae_set=24.0# Set civ's Aggressive Expansion to 24move_capital=75# Move Civ's capital to province ID: 75declare_war=fra# Civ will decalre war on Francedeclare_war2=ger=pol# Germany will declare war on Polandadd_alliance=ger=hun# Civ with tag 'ger'(Germany) will make an alliance with civ with tag 'hun'(Hungary)add_non_aggression=fra=spa# Civ with tag 'fra'(France) will sign Non Aggression Pact with civ with tag 'spa'(Spain)add_military_access=rus=swe# Civ with tag 'rus'(Russia) will get Military Access to civ with tag 'swe'(Sweden)add_guarantee=fra=pol# Civ with tag 'fra'(France) will Guarantee Indpendence of civ with tag 'pol'(Poland)add_defensive_pact=ita=ger# Civ with tag 'ita'(Italy) will sign Defensive Pact with civ with tag 'ger'(Germany)add_truce=ita=ger# Civ with tag 'ita'(Italy) will sign a Truce with civ with tag 'ger'(Germany)change_ideology=4# Civ will change ideology to id: 4change_ideology_civ=2=fra# Civ 'fra' will change ideology to id: 2change_religion=3# Civ will change religion to id: 3change_religion_civ=3=fra# Civ 'fra' will change religion to id: 3# Annexation of provinces ID: 0, 1, 2, 3, 4:annex=0;1;2;3;4;# Annexation of provinces ID: 0, 1, 2, 3, 4, if are controled by "fra", fra is an example and it is the Civ TAG:annex_from_civ=fra=0;1;2;3;4;# ger(Germany) will annex provinces: 646, 652, 647 if are controled by "czsl" (Czechoslovakia)annex_by_civ_from_civ=ger=czsl=646;652;647;annex_by_civ_from_civ=CIVTAG=CIVTAG2=646;652;647;# Annexation of Civilization, fra is an example and it is the Civ TAG:annex_civ=fra# Civilization will be annexed by ger. ger is Civ TAG for Germany.annexed_by_civ=ger# A civilization with the tag "atr" will become a vassal/puppet of the civilization with the tag "ger".make_puppet=ger=atrset_civ_tag=pol# Set Civilization Tag to pol -> Poland# This will change the Civilization to Polandset_civ_tag=spa_n# Set Civilization Tag to spa_n -> Spain, Constitutional Monarchy# This will change the Civilization to Spainset_civ_tag=fra_r# Set Civilization Tag to fra_n -> France, Republic# This will change the Civilization to Franceset_civ_tag2=czsl=slo_z## The civilization with the TAG 'czsl' (Czechoslovakia) will change to 'slo_z' (Slovakia).set_civ_tag2=fra=fra_f## The civilization with the TAG 'fra' (France) will change to 'fra_f' (Vichy France).###### RUN EVENTrun_event=the_austrian_question# The event with ID 'the_austrian_question' will run the next day.# Civilization will get a Random Generaladd_general=true# Civilization will get a General:  Zhukov from the game/characters/Zhukov.json if exists!add_general2=Zhukov# Civilization will get a General:  Galileo from the game/characters/Galileo.json if exists! With 6 Attack, 4 Defense!add_general3=Galileo=6=4# Civilization will get a General:  Galileo from the game/characters/Galileo.json if exists! With random Attack(-1), 3 Defense! -1 to set random valueadd_general3=Galileo=-1=3# Civilization will get Economic Advisor(1):  Franklin from the game/characters/Franklin.json if exists!add_advisor2=1=Franklin# Civilization will get Military Advisor(3):  Zhukov from the game/characters/Zhukov.json if exists!add_advisor2=3=Zhukov# 0 = Administrative Advisor# 1 = Economic Advisor# 2 = Innovation Advisor# 3 = Military AdvisorCivilization will get Random Economic Advisoradd_advisor=1add_ruler=Lukasz=Jakowski=223=23=06=1992# Civilization will get new Ruler:# Name: Lukasz, Surname: Jakowski# Image ID: 223, Will be loaded from: game/rulers/rulersImages/H/223.png for mod it will be: mods/YOUR_MOD_NAME/game/rulers/rulersImages/H/223.png# You can use text instead of an integer for the ruler image. For example, 'Zedong' -> it will load Zedong.png# Born Day: 23, Born Month: 6, Born Year: 1992add_ruler=TheName=TheSurname=Zedong=23=06=1992####### province_economy=15.4 ### A random Civilization province will receive 15.4 economy points.# province_economy_capital=15.4 ### A capital province will receive 15.4 economy points.# province_economy_capital_all=15.4 ### All provinces of Civilization will receive 15.4 economy points.# province_economy_id=332=15.4 ### Province with id 332 will receive 15.4 economy points.province_economy=15.4province_economy_capital=27.2province_economy_all=3.5province_economy_id=11245=4.8province_tax_efficiency=3.2province_tax_efficiency_capital=4.7province_tax_efficiency_all=2.5province_tax_efficiency_id=725=3.0province_manpower=1.2province_manpower_capital=1.7province_manpower_all=3.5province_manpower_id=332=4.0province_growth_rate=8.2province_growth_rate_capital=6.2province_growth_rate_all=7.25province_growth_rate_id=725=3.0province_population=5000province_population_capital=12500province_population_all=2500province_religion=3province_religion_capital=2province_religion_all=3province_religion_id=1000=4province_devastation=27.5province_devastation_capital=45.5province_devastation_all=15province_devastation_id=35=25.0province_unrest=15.5province_unrest_capital=35.5province_unrest_all=25.0province_unrest_id=2541=-25.0province_infrastructure=2province_infrastructure_capital=1province_infrastructure_all=1province_infrastructure_id=332=1province_add_core_civ=332=pol# Province id 332 add core for Polandprovince_remove_core_civ=332=polprovince_remove_core_civ=460=polprovince_add_core_civ=672=poladd_new_army=0=1=3=2# Add new army in capital province,# Unit ID: 0, Army ID: 1# Unit ID: 3, Army ID: 2add_new_army=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=1=6=0=0=0=0=0=0# UnitID=ArmyID# =UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID=UnitID=ArmyID## NOTE SPECIAL# A civilization will get this variable, it can be used to create a chain of events.add_variable=capital_city_economic_development# You can use add_variable=XYz, for example, with an option button to track that this option was selected.# Every time a Civilization runs an event (makes a decision), the 'id' of the event is stored by default in the Civilization's database of variables.# The Civilization with the tag 'ger' (Germany) will receive this variable.add_variable2=ger=sudetenland_accepted#################################################### CIV BONUSES FOR X YEARS ########### We can also set temporary bonuses for X years:bonus_duration=2bonus_monthly_income=24.4# Civilization will have +24.4 monthly income for 2 years.# List of all available, with simple examples:bonus_monthly_income=12.2bonus_monthly_legacy=2.4bonus_monthly_legacy_percentage=10.2bonus_maximum_amount_of_gold=500bonus_tax_efficiency=10.5bonus_growth_rate=15bonus_province_maintenance=-25bonus_buildings_maintenance_cost=-10bonus_maintenance_cost=2.2bonus_production_efficiency=25bonus_income_production=10bonus_income_taxation=7.5bonus_income_economy=10.5bonus_corruption=10.0bonus_inflation=5.0bonus_research=10bonus_research_points=4.5bonus_max_manpower=10000bonus_max_manpower_percentage=10.25bonus_manpower_recovery_speed=25bonus_reinforcement_speed=15bonus_army_morale_recovery=25bonus_army_maintenance=-25bonus_recruitment_time=-15bonus_recruit_army_cost=-20bonus_recruit_army_first_line_cost=-15bonus_recruit_army_second_line_cost=-25bonus_generals_attack=2bonus_generals_defense=1bonus_units_attack=4bonus_units_defense=3bonus_regiments_limit=6bonus_battle_width=4bonus_discipline=7.5bonus_siege_effectiveness=25.5bonus_max_morale=20bonus_army_movement_speed=30.4bonus_manpower_recovery_from_disbanded_army=25bonus_war_score_cost=-25bonus_construction_cost=-10bonus_administration_buildings_cost=-15bonus_military_buildings_cost=-20bonus_economy_buildings_cost=-25bonus_construction_time=-25bonus_invest_in_economy_cost=-13.75bonus_increase_manpower_cost=-7.25bonus_increase_tax_efficiency_cost=-15.4bonus_develop_infrastructure_cost=-12.5bonus_increase_growth_rate_cost=-7.25bonus_diplomacy_points=25bonus_improve_relations_modifier=15bonus_income_from_vassals=25bonus_aggressive_expansion=-25bonus_core_cost=-25bonus_religion_cost=-15bonus_revolutionary_risk=25bonus_loan_interest=-25bonus_loans_limit=1## +1 Max loansbonus_all_characters_life_expectancy=5bonus_advisor_cost=-20bonus_advisors_max_level=2bonus_general_cost=-15bonus_disease_death_rate=75####### CIV BONUSES FOR X YEARS END ##################################################join_alliance_special_id_first_tier=0# Join Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).join_alliance_special_id_second_tier=1# Join Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).leave_alliance_special_id=0# Leave Special Alliance with ID: 0, an alliance like the Holy Roman Empire (HRE).##########kill_ruler=truekill_ruler_chance=35kill_advisor=1# 0 = Administrative Advisor# 1 = Economic Advisor# 2 = Innovation Advisor# 3 = Military Advisorpromote_advisor=2# Promote advisor ID: 2 -> Innovation Advisormilitary_academy=1# Change level of Military Academy: +1military_academy=-2# Change level of Military Academy: -2military_academy_generals=3# Change level of Military Academy for Generals: +3capital_city_level=1# Change level of Capital City Level: +1supreme_court=2# Change level of Supreme Court: +2nuclear_reactor=1# Change level of Nuclear Reactor: +1############explode=ming# Ming will split into multiple civilizations based on the number of civilizations present within Ming's provinces that have cores and no provinces############## Change price of resource random Resourceprice_change_random=15=16=12=36### Change price of random Resource, by:# 15% min + Random(16)%, price will be randomly decreased or increased# For 12 months + Random(36) Months--------------------price_change_random=5=35=24=36## Change price of random Resource, by:# 5% min + Random(35)%, price will be randomly decreased or increased# For 24 months + Random(36) Months--------------------price_change_random_up=6=24=21=0## Change price of random Resource, by:# 6% min + Random(24)%, price will be Increased!# For 21 months + Random(0) Months--------------------price_change_random_down=40=25=12=4## Change price of random Resource, by:# 40% min + Random(25)%, price will be Decreased!# For 12 months + Random(4) Months############## Change price of resource Resource IDprice_change=0=10=15=12=24## Change price of ResourceID: 0 -> Grain, by:# 10% min + Random(15)%, price will be randomly decreased or increased# For 12 months + Random(24) Months## ResourceID can be find in: game/resources/Resources.json## ID: 0 will be Grain--------------------price_change=9=20=12=96=12## Change price of ResourceID: 9 -> Cocoa, by:# 20% min + Random(12)%, price will be randomly decreased or increased# For 96 months + Random(12) Months## ResourceID can be find in: game/resources/Resources.json## ID: 9 will be Cocoa--------------------price_change_up=0=20=40=18=24## Change price of ResourceID: 0 -> Grain, by:# 20% min + Random(40)%, price will be Increased!# For 18 months + Random(24) Months## ResourceID can be find in: game/resources/Resources.json## ID: 0 will be Grain--------------------price_change_down=0=20=40=18=24## Change price of ResourceID: 0 -> Grain, by:# 20% min + Random(40)%, price will be Decreased!# For 18 months + Random(24) Months## ResourceID can be find in: game/resources/Resources.json## ID: 0 will be Grain--------------------price_change_group=0=15=5=24=12## Change price of Resources with GroupID: 0 -> Food:# 15% min + Random(5)%, price will be randomly decreased or increased# For 24 months + Random(12) Months## GroupID can be find in: game/resources/Resources.jsonprice_change_group_up=3=15=5=24=12## Change price of Resources with GroupID: 3 -> Luxury:# 15% min + Random(5)%, price will be Increased!price_change_group_down=4=15=5=24=12## Change price of Resources with GroupID: 4 -> Production Resources:1# 15% min + Random(5)%, price will be Decreased!Group ID:0 -> Food1 -> Commodities2 -> Luxury Commodities3 -> Luxury4 -> Production Resources################################////////////////////////////////////////#### FULL CODE OF THE EXAMPLE EVENT ####////////////////////////////////////////id=economys_golden_milestonetitle=economys_golden_milestone.tdesc=economys_golden_milestone.dmission_desc=economys_golden_milestone.mimage=economy.pngshow_in_missions=truemission_image=0only_once=truetrigger_andnext_andinvested_in_economy_over=1trigger_and_endoption_btnname=GoodNewsai=20gold=250legacy=25bonus_duration=1bonus_monthly_income=27.4option_endoption_btnname=EconomicTriumphai=10province_economy_capital=10legacy=50bonus_duration=2bonus_monthly_income=38.4option_end# Files list_common.txt, list_global.txt, list_siege.txt are only for Mobile Devices!'''
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


    prompt = f"{effects} \n Convert the following event according to the given triggers and effects, adapting it as closely as possible:\n\n{event_str}\n\nAdapted Event:\n"
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
