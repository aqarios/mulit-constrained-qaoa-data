Minimize
 obj: + 126.7 x_user1_load0_hour0 + 135.7 x_user1_load0_hour1 
 + 140.6 x_user1_load0_hour2 + 157.7 x_user1_load0_hour3 
 + 166.7 x_user1_load0_hour4 + 173.6 x_user1_load0_hour5 
 + 176.7 x_user1_load0_hour6 + 189.7 x_user1_load0_hour7 
 + 204.6 x_user1_load0_hour8 + 213.7 x_user1_load0_hour9 
 + 130.8 x_user1_load1_hour0 + 130.6 x_user1_load1_hour1 
 + 142.6 x_user1_load1_hour2 + 156.8 x_user1_load1_hour3 
 + 170.6 x_user1_load1_hour4 + 170.6 x_user1_load1_hour5 
 + 174.8 x_user1_load1_hour6 + 192.6 x_user1_load1_hour7 
 + 202.6 x_user1_load1_hour8 + 214.8 x_user1_load1_hour9 
 + 226.6 x_user1_load1_hour10 + 145.3 x_user1_load2_hour0 
 + 162.1 x_user1_load2_hour1 + 163.6 x_user1_load2_hour2 
 + 185.3 x_user1_load2_hour3 + 192.1 x_user1_load2_hour4 
 + 204.6 x_user1_load2_hour5 + 208.3 x_user1_load2_hour6 
 + 220.1 x_user1_load2_hour7 + 240.6 x_user1_load2_hour8 
 + 249.3 x_user1_load2_hour9 + 82.7 x_user1_load3_hour0 
 + 89.4 x_user1_load3_hour1 + 88.9 x_user1_load3_hour2 
 + 101.7 x_user1_load3_hour3 + 109.4 x_user1_load3_hour4 
 + 115.9 x_user1_load3_hour5 + 113.7 x_user1_load3_hour6 
 + 122.4 x_user1_load3_hour7 + 133.9 x_user1_load3_hour8 
 + 138.7 x_user1_load3_hour9 + 148.4 x_user1_load3_hour10 
 + 149 x_user1_load4_hour0 + 163.8 x_user1_load4_hour1 
 + 166.2 x_user1_load4_hour2 + 184 x_user1_load4_hour3 
 + 193.8 x_user1_load4_hour4 + 208.2 x_user1_load4_hour5 
 + 209 x_user1_load4_hour6 + 222.8 x_user1_load4_hour7 
 + 242.2 x_user1_load4_hour8 + 60 x_user1_load5_hour0 
 + 68.1 x_user1_load5_hour1 + 63.9 x_user1_load5_hour2 
 + 75 x_user1_load5_hour3 + 80.1 x_user1_load5_hour4 
 + 87.9 x_user1_load5_hour5 + 84 x_user1_load5_hour6 
 + 89.1 x_user1_load5_hour7 + 99.9 x_user1_load5_hour8 
 + 102 x_user1_load5_hour9 + 110.1 x_user1_load5_hour10 
 + 114.9 x_user1_load5_hour11 + 108.1 x_user1_load6_hour0 
 + 109.3 x_user1_load6_hour1 + 117.6 x_user1_load6_hour2 
 + 130.1 x_user1_load6_hour3 + 141.3 x_user1_load6_hour4 
 + 142.6 x_user1_load6_hour5 + 145.1 x_user1_load6_hour6 
 + 159.3 x_user1_load6_hour7 + 168.6 x_user1_load6_hour8 
 + 178.1 x_user1_load6_hour9 + 188.3 x_user1_load6_hour10 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7 
 + 1 x_user1_load0_hour8 + 1 x_user1_load0_hour9  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8 + 1 x_user1_load1_hour9 + 1 x_user1_load1_hour10 
  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7 
 + 1 x_user1_load2_hour8 + 1 x_user1_load2_hour9  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour9 + 1 x_user1_load3_hour10 
  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8  = 1.0
 load_load5_scheduled: + 1 x_user1_load5_hour0 + 1 x_user1_load5_hour1 
 + 1 x_user1_load5_hour2 + 1 x_user1_load5_hour3 + 1 x_user1_load5_hour4 
 + 1 x_user1_load5_hour5 + 1 x_user1_load5_hour6 + 1 x_user1_load5_hour7 
 + 1 x_user1_load5_hour8 + 1 x_user1_load5_hour9 + 1 x_user1_load5_hour10 
 + 1 x_user1_load5_hour11  = 1.0
 load_load6_scheduled: + 1 x_user1_load6_hour0 + 1 x_user1_load6_hour1 
 + 1 x_user1_load6_hour2 + 1 x_user1_load6_hour3 + 1 x_user1_load6_hour4 
 + 1 x_user1_load6_hour5 + 1 x_user1_load6_hour6 + 1 x_user1_load6_hour7 
 + 1 x_user1_load6_hour8 + 1 x_user1_load6_hour9 + 1 x_user1_load6_hour10 
  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 4 x_user1_load2_hour0 + 3 x_user1_load3_hour0 + 4 x_user1_load4_hour0 
 + 3 x_user1_load5_hour0 + 2 x_user1_load6_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 4 x_user1_load1_hour0 + 4 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 3 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 4 x_user1_load4_hour1 + 1 x_user1_load4_hour0 + 3 x_user1_load5_hour1 
 + 2 x_user1_load6_hour1 + 3 x_user1_load6_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 4 x_user1_load1_hour1 
 + 4 x_user1_load2_hour2 + 1 x_user1_load2_hour1 + 2 x_user1_load2_hour0 
 + 3 x_user1_load3_hour2 + 1 x_user1_load3_hour1 + 4 x_user1_load4_hour2 
 + 1 x_user1_load4_hour1 + 1 x_user1_load4_hour0 + 3 x_user1_load5_hour2 
 + 2 x_user1_load6_hour2 + 3 x_user1_load6_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 4 x_user1_load1_hour2 
 + 4 x_user1_load2_hour3 + 1 x_user1_load2_hour2 + 2 x_user1_load2_hour1 
 + 3 x_user1_load3_hour3 + 1 x_user1_load3_hour2 + 4 x_user1_load4_hour3 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour1 + 1 x_user1_load4_hour0 
 + 3 x_user1_load5_hour3 + 2 x_user1_load6_hour3 + 3 x_user1_load6_hour2 
  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 4 x_user1_load1_hour3 
 + 4 x_user1_load2_hour4 + 1 x_user1_load2_hour3 + 2 x_user1_load2_hour2 
 + 3 x_user1_load3_hour4 + 1 x_user1_load3_hour3 + 4 x_user1_load4_hour4 
 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour1 
 + 3 x_user1_load5_hour4 + 2 x_user1_load6_hour4 + 3 x_user1_load6_hour3 
  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 4 x_user1_load1_hour4 
 + 4 x_user1_load2_hour5 + 1 x_user1_load2_hour4 + 2 x_user1_load2_hour3 
 + 3 x_user1_load3_hour5 + 1 x_user1_load3_hour4 + 4 x_user1_load4_hour5 
 + 1 x_user1_load4_hour4 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour2 
 + 3 x_user1_load5_hour5 + 2 x_user1_load6_hour5 + 3 x_user1_load6_hour4 
  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 2 x_user1_load0_hour5 
 + 1 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 4 x_user1_load1_hour5 
 + 4 x_user1_load2_hour6 + 1 x_user1_load2_hour5 + 2 x_user1_load2_hour4 
 + 3 x_user1_load3_hour6 + 1 x_user1_load3_hour5 + 4 x_user1_load4_hour6 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour4 + 1 x_user1_load4_hour3 
 + 3 x_user1_load5_hour6 + 2 x_user1_load6_hour6 + 3 x_user1_load6_hour5 
  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 1 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 4 x_user1_load1_hour6 
 + 4 x_user1_load2_hour7 + 1 x_user1_load2_hour6 + 2 x_user1_load2_hour5 
 + 3 x_user1_load3_hour7 + 1 x_user1_load3_hour6 + 4 x_user1_load4_hour7 
 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour4 
 + 3 x_user1_load5_hour7 + 2 x_user1_load6_hour7 + 3 x_user1_load6_hour6 
  <= 4.0
 MaxLoadConnection_user1_8: + 3 x_user1_load0_hour8 + 2 x_user1_load0_hour7 
 + 1 x_user1_load0_hour6 + 2 x_user1_load1_hour8 + 4 x_user1_load1_hour7 
 + 4 x_user1_load2_hour8 + 1 x_user1_load2_hour7 + 2 x_user1_load2_hour6 
 + 3 x_user1_load3_hour8 + 1 x_user1_load3_hour7 + 4 x_user1_load4_hour8 
 + 1 x_user1_load4_hour7 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour5 
 + 3 x_user1_load5_hour8 + 2 x_user1_load6_hour8 + 3 x_user1_load6_hour7 
  <= 4.0
 MaxLoadConnection_user1_9: + 3 x_user1_load0_hour9 + 2 x_user1_load0_hour8 
 + 1 x_user1_load0_hour7 + 2 x_user1_load1_hour9 + 4 x_user1_load1_hour8 
 + 4 x_user1_load2_hour9 + 1 x_user1_load2_hour8 + 2 x_user1_load2_hour7 
 + 3 x_user1_load3_hour9 + 1 x_user1_load3_hour8 + 1 x_user1_load4_hour8 
 + 1 x_user1_load4_hour7 + 1 x_user1_load4_hour6 + 3 x_user1_load5_hour9 
 + 2 x_user1_load6_hour9 + 3 x_user1_load6_hour8  <= 4.0
 MaxLoadConnection_user1_10: + 2 x_user1_load0_hour9 + 1 x_user1_load0_hour8 
 + 2 x_user1_load1_hour10 + 4 x_user1_load1_hour9 + 1 x_user1_load2_hour9 
 + 2 x_user1_load2_hour8 + 3 x_user1_load3_hour10 + 1 x_user1_load3_hour9 
 + 1 x_user1_load4_hour8 + 1 x_user1_load4_hour7 + 3 x_user1_load5_hour10 
 + 2 x_user1_load6_hour10 + 3 x_user1_load6_hour9  <= 4.0
 MaxLoadConnection_user1_11: + 1 x_user1_load0_hour9 + 4 x_user1_load1_hour10 
 + 2 x_user1_load2_hour9 + 1 x_user1_load3_hour10 + 1 x_user1_load4_hour8 
 + 3 x_user1_load5_hour11 + 3 x_user1_load6_hour10  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load0_hour8
  x_user1_load0_hour9 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load1_hour4
  x_user1_load1_hour5 x_user1_load1_hour6 x_user1_load1_hour7
  x_user1_load1_hour8 x_user1_load1_hour9 x_user1_load1_hour10
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4 x_user1_load2_hour5
  x_user1_load2_hour6 x_user1_load2_hour7 x_user1_load2_hour8
  x_user1_load2_hour9 x_user1_load3_hour0 x_user1_load3_hour1
  x_user1_load3_hour2 x_user1_load3_hour3 x_user1_load3_hour4
  x_user1_load3_hour5 x_user1_load3_hour6 x_user1_load3_hour7
  x_user1_load3_hour8 x_user1_load3_hour9 x_user1_load3_hour10
  x_user1_load4_hour0 x_user1_load4_hour1 x_user1_load4_hour2
  x_user1_load4_hour3 x_user1_load4_hour4 x_user1_load4_hour5
  x_user1_load4_hour6 x_user1_load4_hour7 x_user1_load4_hour8
  x_user1_load5_hour0 x_user1_load5_hour1 x_user1_load5_hour2
  x_user1_load5_hour3 x_user1_load5_hour4 x_user1_load5_hour5
  x_user1_load5_hour6 x_user1_load5_hour7 x_user1_load5_hour8
  x_user1_load5_hour9 x_user1_load5_hour10 x_user1_load5_hour11
  x_user1_load6_hour0 x_user1_load6_hour1 x_user1_load6_hour2
  x_user1_load6_hour3 x_user1_load6_hour4 x_user1_load6_hour5
  x_user1_load6_hour6 x_user1_load6_hour7 x_user1_load6_hour8
  x_user1_load6_hour9 x_user1_load6_hour10
General

End