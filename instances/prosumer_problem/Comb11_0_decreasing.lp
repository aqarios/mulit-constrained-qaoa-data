Minimize
 obj: + 233.3 x_user1_load0_hour0 + 222.3 x_user1_load0_hour1 
 + 207.4 x_user1_load0_hour2 + 204.3 x_user1_load0_hour3 
 + 193.3 x_user1_load0_hour4 + 180.4 x_user1_load0_hour5 
 + 163.3 x_user1_load0_hour6 + 156.3 x_user1_load0_hour7 
 + 151.4 x_user1_load0_hour8 + 140.3 x_user1_load0_hour9 
 + 237.2 x_user1_load1_hour0 + 217.4 x_user1_load1_hour1 
 + 209.4 x_user1_load1_hour2 + 203.2 x_user1_load1_hour3 
 + 197.4 x_user1_load1_hour4 + 177.4 x_user1_load1_hour5 
 + 161.2 x_user1_load1_hour6 + 159.4 x_user1_load1_hour7 
 + 149.4 x_user1_load1_hour8 + 141.2 x_user1_load1_hour9 
 + 133.4 x_user1_load1_hour10 + 268.7 x_user1_load2_hour0 
 + 261.9 x_user1_load2_hour1 + 240.4 x_user1_load2_hour2 
 + 238.7 x_user1_load2_hour3 + 221.9 x_user1_load2_hour4 
 + 211.4 x_user1_load2_hour5 + 191.7 x_user1_load2_hour6 
 + 179.9 x_user1_load2_hour7 + 177.4 x_user1_load2_hour8 
 + 162.7 x_user1_load2_hour9 + 159.3 x_user1_load3_hour0 
 + 152.6 x_user1_load3_hour1 + 139.1 x_user1_load3_hour2 
 + 138.3 x_user1_load3_hour3 + 132.6 x_user1_load3_hour4 
 + 126.1 x_user1_load3_hour5 + 110.3 x_user1_load3_hour6 
 + 105.6 x_user1_load3_hour7 + 104.1 x_user1_load3_hour8 
 + 95.3 x_user1_load3_hour9 + 91.6 x_user1_load3_hour10 
 + 269 x_user1_load4_hour0 + 260.2 x_user1_load4_hour1 
 + 239.8 x_user1_load4_hour2 + 234 x_user1_load4_hour3 
 + 220.2 x_user1_load4_hour4 + 211.8 x_user1_load4_hour5 
 + 189 x_user1_load4_hour6 + 179.2 x_user1_load4_hour7 
 + 175.8 x_user1_load4_hour8 + 120 x_user1_load5_hour0 
 + 117.9 x_user1_load5_hour1 + 104.1 x_user1_load5_hour2 
 + 105 x_user1_load5_hour3 + 99.9 x_user1_load5_hour4 
 + 98.1 x_user1_load5_hour5 + 84 x_user1_load5_hour6 
 + 78.9 x_user1_load5_hour7 + 80.1 x_user1_load5_hour8 
 + 72 x_user1_load5_hour9 + 69.9 x_user1_load5_hour10 
 + 65.1 x_user1_load5_hour11 + 197.9 x_user1_load6_hour0 
 + 182.7 x_user1_load6_hour1 + 174.4 x_user1_load6_hour2 
 + 169.9 x_user1_load6_hour3 + 164.7 x_user1_load6_hour4 
 + 149.4 x_user1_load6_hour5 + 134.9 x_user1_load6_hour6 
 + 132.7 x_user1_load6_hour7 + 125.4 x_user1_load6_hour8 
 + 117.9 x_user1_load6_hour9 + 111.7 x_user1_load6_hour10 

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