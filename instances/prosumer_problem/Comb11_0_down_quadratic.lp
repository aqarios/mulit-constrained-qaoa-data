Minimize
 obj: + 143.3 x_user1_load0_hour0 + 174.5 x_user1_load0_hour1 
 + 195.1 x_user1_load0_hour2 + 221 x_user1_load0_hour3 
 + 232.2 x_user1_load0_hour4 + 234.6 x_user1_load0_hour5 
 + 226.6 x_user1_load0_hour6 + 221.8 x_user1_load0_hour7 
 + 212.5 x_user1_load0_hour8 + 190.3 x_user1_load0_hour9 
 + 148.4 x_user1_load1_hour0 + 170.6 x_user1_load1_hour1 
 + 198.2 x_user1_load1_hour2 + 221.2 x_user1_load1_hour3 
 + 237.2 x_user1_load1_hour4 + 232.8 x_user1_load1_hour5 
 + 225.6 x_user1_load1_hour6 + 226 x_user1_load1_hour7 
 + 211.6 x_user1_load1_hour8 + 192.4 x_user1_load1_hour9 
 + 166.6 x_user1_load1_hour10 + 165.3 x_user1_load2_hour0 
 + 207.5 x_user1_load2_hour1 + 227 x_user1_load2_hour2 
 + 258.6 x_user1_load2_hour3 + 267.6 x_user1_load2_hour4 
 + 274.4 x_user1_load2_hour5 + 265 x_user1_load2_hour6 
 + 255.4 x_user1_load2_hour7 + 247.4 x_user1_load2_hour8 
 + 219.3 x_user1_load2_hour9 + 87.1 x_user1_load3_hour0 
 + 110.4 x_user1_load3_hour1 + 122.3 x_user1_load3_hour2 
 + 142.8 x_user1_load3_hour3 + 153.8 x_user1_load3_hour4 
 + 159.2 x_user1_load3_hour5 + 151.4 x_user1_load3_hour6 
 + 150 x_user1_load3_hour7 + 147.4 x_user1_load3_hour8 
 + 133.1 x_user1_load3_hour9 + 119.4 x_user1_load3_hour10 
 + 171.2 x_user1_load4_hour0 + 210.3 x_user1_load4_hour1 
 + 229.6 x_user1_load4_hour2 + 256.2 x_user1_load4_hour3 
 + 267 x_user1_load4_hour4 + 274.8 x_user1_load4_hour5 
 + 261.2 x_user1_load4_hour6 + 252.5 x_user1_load4_hour7 
 + 242.4 x_user1_load4_hour8 + 60 x_user1_load5_hour0 
 + 81.3 x_user1_load5_hour1 + 87.3 x_user1_load5_hour2 
 + 105 x_user1_load5_hour3 + 113.4 x_user1_load5_hour4 
 + 121.2 x_user1_load5_hour5 + 114 x_user1_load5_hour6 
 + 112.2 x_user1_load5_hour7 + 113.4 x_user1_load5_hour8 
 + 102 x_user1_load5_hour9 + 93.3 x_user1_load5_hour10 
 + 78.3 x_user1_load5_hour11 + 121.3 x_user1_load6_hour0 
 + 141.5 x_user1_load6_hour1 + 163.2 x_user1_load6_hour2 
 + 183.4 x_user1_load6_hour3 + 196.8 x_user1_load6_hour4 
 + 194.8 x_user1_load6_hour5 + 188.2 x_user1_load6_hour6 
 + 188.2 x_user1_load6_hour7 + 177.6 x_user1_load6_hour8 
 + 161.3 x_user1_load6_hour9 + 140.5 x_user1_load6_hour10 

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