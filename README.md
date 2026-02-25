# MLB-Pitch-Prediction
Milestone 1

For the semester-long project, I wanted to mix my passion for baseball with pitch analysis and chose to do the project on using machine learning to predict the next pitch by a MLB pitcher. 

Below I attached the links to the websites and Google Colab notebooks I used to download the referenced data. I drew information from the Baseball Savant website including statistics on pitcher arsenal stats for both batters and pitchers, pitcher arsenal, pitcher pitch tempo, year to year statical changes for pitchers based on in-the-zone pitchers and percentage, swing percentage, WHIFF percentage, and xwOBA statistics. Most of the bulk of data came from the Pybaseball PYPI package. 

The columns in the PYPI package includes, pitch_type, game_date, release_speed, release_pos_x, release_pos_z, player_name, batter, pitcher, events, description, spin_dir, spin_rate_deprecated, break_angle_deprecated, break_length_deprecated, zone, des, game_type, stand, p_throws, home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, pfx_x, pfx_z, plate_x, plate_z, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, hc_x, hc_y, tfs_deprecated, tfs_zulu_deprecated, fielder_2, umpire, sv_id, vx0, vy0, vz0, ax, ay, az, sz_top, sz_bot, hit_distance_sc, launch_speed, launch_angle, effective_speed, release_spin_rate, release_extension, game_pk, pitcher.1, fielder_2.1, fielder_3, fielder_4, fielder_5, fielder_6, fielder_7, fielder_8, fielder_9, release_pos_y, estimated_ba_using_speedangle, estimated_woba_using_speedangle, woba_value, woba_denom, babip_value, iso_value, launch_speed_angle, at_bat_number, pitch_number, pitch_name, home_score, away_score, bat_score, fld_score, post_away_score, post_home_score, post_bat_score, post_fld_score, if_fielding_alignment, of_fielding_alignment, spin_axis, delta_home_win_exp, delta_run_exp, bat_speed, and swing_length.

The Baseball Savant databases have columns, like for the pitcher arsenal stats, being: last_name, first_name, player_id, team_name_alt, pitch_type, pitch_name, run_value_per_100, run_value, pitches, pitch_usage, pa, ba, slg, woba, whiff_percent, k_percent, put_away, est_ba, est_slg, est_woba, and hard_hit_percent. 

I will use these databases to get overall pitcher and batter statistics from Baseball Savant, with past matchups and pitching sequences through the Pybaseball package. I will use a logistic regression model to have a categorical outcome, being the pitch type. 

Milestone 2

For milestone 2 I used a Google Virtual Machine Instance to download Python, set up a Python development environment, and download the dependencies such as pybaseball. After doing so I used the command nano to create a Python program and copy the codes I used in my Google Colab notebook from milestone 1. I divided the 20 MLB seasons into 4 programs with 5 seasons each. The codes for these Python programs are attached in Appendix A. 

After doing so, I used the code, gcloud auth login, to attach the instance to my Google project. I created a Google Cloud bucket named my-project-bucket-jl. Finally, I uploaded all the .csv files to my bucket using the code: gcloud storage cp --recursive *.csv  gs://my-project-bucket-jl/landing/.

Milestone 3

During the exploratory data analysis stage, I was surprised by the scale of data after all the dataframes were merged. Together the dataframe has a shape of 14299659 rows and 94 columns. There were 2054373 NA values for the pitch_type column. Therefore, I would have to drop those rows because the focus is on the pitch_type, and NA values would be redundant. 

The histograms were very insightful, especially the ‘pitch_type’ histogram. Out of all the pitch types the most frequently used pitch was four-seam fastball, which was very interesting to learn. It didn’t feel so surprising, given how that is a common pitch to hear was used or mentioned during games, but to see how it was almost tripled the amount as the second most used pitch type, sinker pitch, was still pretty surprising. 

Also, the histograms of the differences in right and left-handed pitchers and batters was fun to see. There’s always the conversation on whether to bat certain players based on their batting hand. The result ended up being that there were for both, the majority of players are right-handed either in batting stance or for pitching arm. 

For the clean data, I dropped the columns that didn’t have a direct relation with the next pitch, including fielder numbers, bat swings, etc. 

I had trouble finding the correct pathway to loop through all the MLB database files. However, it was nice to see the dataframes being cleaned through the loop. Ultimately, I struggled with uploading the parquet files into the Google Cloud storage bucket folder. 

Milestone 4

The main goal for the model is to predict the next pitch of a pitcher based on the features given. For milestone 4, the code begins with importing libraries and modules. Then, it is the how to access the files that are from the cleaned folder. I merged all the files to have a merged database to work with. The table below reflects the decisions made early on to index, encode, and assemble the columns. For ‘events’, to simplify the type to whether the bat continues, the batter is out, or if the batter got on base, a user-defined function was used. Then that too was indexed. 

The code then splits for the training and testing data, and the logistical regression, lr, variable. Together they make the variables for the pitch_type_pipe. The code then has a grid in order to perform the cross-validater. Finally, there were some evaluation methods used on the results and showed how to save the best model to the Google Cloud storage.

One of the main challenges I had was the volume of the indexing for some columns, such as the “one_base” ones as each id would make the index a very large quantity for the nodes used. I chose to change it to simply reflect if there was a runner on base or not. Another challenger was indexing pitch_type as I wasn’t sure if I should index each type, however, the codes reflect the “one vs all” method in which the “one” is the most frequently used pitch which was the Four Seam Fastball, known in the data as “FF.”

For the outputs, the AUC value of my data is 0.7961624286122574. For accuracy it equals 0.2575624161948243, precision equals 0.3800459875394758, recall equals 0.12575115899867564, and F1 score equals 0.18897387513150202. The ROC curve is attached below. The best model has an intercept of -0.20270887673343196. The confusion matrix is also attached along with the hyper parameters of the best model. 

Milestone 5

The first visual, “Number of FF Pitches Over the Years,” is a line graph to document the frequency of the Four-Seam Fastball throughout the years. As new pitch types have grown in popularity and usage, I thought it would be important to document how a staple pitch like the Four-Seam Fastball has continued to be used throughout the years. It does show, especially in recent years, many spikes and drops, which is interesting especially as new regulations, like the pitch clocks, are being used now. However, it does show, even with ups and downs, a daily consistently high usage throughout the years. 

The second visual, ”Number of FF Pitches Based on Pitch Zone,” is a bar graph that visualizes the frequency of the Four-Seam Fastball in every zone. There is a gap for the value 10 because MLB skips 10 when designating the pitch zone. There is an increase of Four-Seam Fastball in out of strike zone values, 11-14. Especially the upper out of strike corners, 11 and 12, experience a higher usage of the Four-Seam Fastball. This could be because the nature of the fastball is focused on speed rather than precision; the goal is mainly to get the batter to swing and miss the ball because of its speed.

The third visual, “Number of FF Pitches Based on Runners on Base,” visualizes the frequency of the Four-Seam Fastball for every scenario of runners on base using a bar graph. No Runners have the most, and that could be because it is a more common occurrence throughout the game, as each inning starts with no runners on base. Also, as bases are filled up, some pitchers could use different pitch types in their arsenal for a different strategy. 

The fourth visual, “Number of FF Pitches Based on the Count,” is a bar graph that visualizes the frequency of the Four-Seam Fastball for every count possibility. Similar to the graph “FF_by_runners_on_base,” having an empty count is a more common situation as every batter starts with an empty count. Also, similar to “FF_by_runners_on_base,” pitchers may turn to other pitch types as the count grows, to get the batter out. However, it does stay pretty consistent, which also shows how it is a more commonly used pitch for many situations. 

The fifth visual, “Number of FF pitches Over the Innings,” visualizes the frequency of the Four-Seam Fastball through all the innings in a game. Since a baseball game is usually 9 innings there is a start drop once the value for innings reaches 10 and beyond, as there is less data. However, after the drop, there is a consistent usage of the Four-Seam Fastball, even though it is a lot less. The graph “Number of FF Pitches Over 9 Innings” also goes more in-depth on the Four-Seam Fastball frequency within the first 9 innings. 

The sixth visual, “Number of FF Pitches Over 9 Innings,” is a line graph to document the frequency of the Four-Seam Fastball through nine innings, as that is the usual baseball game length. Based on the graph there is a very high number of Four-Seam Fastball usage in the first innings, then a dip during the middle of the game, and then another rise at the end before it drops again for the ninth inning. Pitchers at the beginning of the game usually try not to show batters their whole pitching arsenal, especially as starting pitchers will have to see the same batters around three more times. Therefore, a staple pitch like the Four-Seam Fastball is a usual option in the first innings, which could explain why it has such a peak at the beginning of the game. 

Milestone 6

The project's main objective was to create a machine-learning model using logistic regression to determine the pitcher's next pitch type in an MLB game. Sourcing data through the pybaseball Python package, I used the information gathered to clean them and perform feature engineering. The codes reflect a “one vs all” method in determining if the next pitch by the pitcher was the most popular MLB pitch, the Four Seam Fastball, also known in the data as “FF.” 

Although the project did not result in the most accurate outcome, it does help create a starting point to explore more and understand pitching patterns within the league. The visuals used help paint the story and idea of these patterns as pitch usage reflects the pitching patterns of a typical MLB game. 

The feature importance data also reflects where the next steps in this project may be, such as what features should be weighted more versus others. Based on my best model’s feature importance, using Random Forest, the batter ID, the zone, and whether a player was on 3B tended to be the most important. The result also reflects the visuals, as pitchers pitched the FF less frequently with runners on 3B. Also, it can be assumed that a pitcher will change his game plan depending on the batter, as a pitcher wouldn’t pitch the same way to every batter. 

In conclusion, the project is a starting point for constructing and understanding patterns between pitchers and pitch types, especially as there are so many more pitch types to examine and features to be weighted. 
