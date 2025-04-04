Minimize
 obj: + 212.8 x_user1_load0_hour0 + 175 x_user1_load0_hour1 
 + 142.8 x_user1_load0_hour2 + 132.2 x_user1_load0_hour3 
 + 123.2 x_user1_load0_hour4 + 121.8 x_user1_load0_hour5 
 + 126 x_user1_load0_hour6 + 149.8 x_user1_load0_hour7 
 + 185.2 x_user1_load0_hour8 + 240.4 x_user1_load1_hour0 
 + 191.6 x_user1_load1_hour1 + 164 x_user1_load1_hour2 
 + 150.6 x_user1_load1_hour3 + 144.4 x_user1_load1_hour4 
 + 138.4 x_user1_load1_hour5 + 149.6 x_user1_load1_hour6 
 + 183 x_user1_load1_hour7 + 221.6 x_user1_load1_hour8 
 + 254.6 x_user1_load2_hour0 + 219.4 x_user1_load2_hour1 
 + 182 x_user1_load2_hour2 + 174.4 x_user1_load2_hour3 
 + 158.6 x_user1_load2_hour4 + 166.6 x_user1_load2_hour5 
 + 186.4 x_user1_load2_hour6 + 218 x_user1_load2_hour7 
 + 273.4 x_user1_load2_hour8 + 139 x_user1_load3_hour0 
 + 116 x_user1_load3_hour1 + 94.4 x_user1_load3_hour2 
 + 88.2 x_user1_load3_hour3 + 81.4 x_user1_load3_hour4 
 + 82 x_user1_load3_hour5 + 86 x_user1_load3_hour6 + 101.4 x_user1_load3_hour7 
 + 126.2 x_user1_load3_hour8 + 107.6 x_user1_load4_hour0 
 + 84.2 x_user1_load4_hour1 + 71.6 x_user1_load4_hour2 
 + 64.8 x_user1_load4_hour3 + 62.8 x_user1_load4_hour4 
 + 58.6 x_user1_load4_hour5 + 61.2 x_user1_load4_hour6 
 + 75.6 x_user1_load4_hour7 + 90.8 x_user1_load4_hour8 
 + 111.8 x_user1_load4_hour9 + 153.8 x_user1_load5_hour0 
 + 126.6 x_user1_load5_hour1 + 98.8 x_user1_load5_hour2 
 + 90.4 x_user1_load5_hour3 + 83.4 x_user1_load5_hour4 
 + 81.8 x_user1_load5_hour5 + 77.6 x_user1_load5_hour6 
 + 90.8 x_user1_load5_hour7 + 113.4 x_user1_load5_hour8 
 + 135.4 x_user1_load5_hour9 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7 
 + 1 x_user1_load0_hour8  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7 
 + 1 x_user1_load2_hour8  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour8  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8 + 1 x_user1_load4_hour9  = 1.0
 load_load5_scheduled: + 1 x_user1_load5_hour0 + 1 x_user1_load5_hour1 
 + 1 x_user1_load5_hour2 + 1 x_user1_load5_hour3 + 1 x_user1_load5_hour4 
 + 1 x_user1_load5_hour5 + 1 x_user1_load5_hour6 + 1 x_user1_load5_hour7 
 + 1 x_user1_load5_hour8 + 1 x_user1_load5_hour9  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0 + 2 x_user1_load3_hour0 + 1 x_user1_load4_hour0 
 + 3 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 4 x_user1_load1_hour0 + 3 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 2 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour1 + 2 x_user1_load4_hour0 + 3 x_user1_load5_hour1 
 + 1 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 4 x_user1_load1_hour1 
 + 1 x_user1_load1_hour0 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 
 + 4 x_user1_load2_hour0 + 2 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour0 + 1 x_user1_load4_hour2 + 2 x_user1_load4_hour1 
 + 3 x_user1_load5_hour2 + 1 x_user1_load5_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 4 x_user1_load1_hour2 
 + 1 x_user1_load1_hour1 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 
 + 4 x_user1_load2_hour1 + 2 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
 + 1 x_user1_load3_hour1 + 1 x_user1_load4_hour3 + 2 x_user1_load4_hour2 
 + 3 x_user1_load5_hour3 + 1 x_user1_load5_hour2  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 4 x_user1_load1_hour3 
 + 1 x_user1_load1_hour2 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 
 + 4 x_user1_load2_hour2 + 2 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
 + 1 x_user1_load3_hour2 + 1 x_user1_load4_hour4 + 2 x_user1_load4_hour3 
 + 3 x_user1_load5_hour4 + 1 x_user1_load5_hour3  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 4 x_user1_load1_hour4 
 + 1 x_user1_load1_hour3 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 
 + 4 x_user1_load2_hour3 + 2 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour3 + 1 x_user1_load4_hour5 + 2 x_user1_load4_hour4 
 + 3 x_user1_load5_hour5 + 1 x_user1_load5_hour4  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 2 x_user1_load0_hour5 
 + 1 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 4 x_user1_load1_hour5 
 + 1 x_user1_load1_hour4 + 3 x_user1_load2_hour6 + 1 x_user1_load2_hour5 
 + 4 x_user1_load2_hour4 + 2 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
 + 1 x_user1_load3_hour4 + 1 x_user1_load4_hour6 + 2 x_user1_load4_hour5 
 + 3 x_user1_load5_hour6 + 1 x_user1_load5_hour5  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 1 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 4 x_user1_load1_hour6 
 + 1 x_user1_load1_hour5 + 3 x_user1_load2_hour7 + 1 x_user1_load2_hour6 
 + 4 x_user1_load2_hour5 + 2 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 1 x_user1_load3_hour5 + 1 x_user1_load4_hour7 + 2 x_user1_load4_hour6 
 + 3 x_user1_load5_hour7 + 1 x_user1_load5_hour6  <= 4.0
 MaxLoadConnection_user1_8: + 3 x_user1_load0_hour8 + 2 x_user1_load0_hour7 
 + 1 x_user1_load0_hour6 + 2 x_user1_load1_hour8 + 4 x_user1_load1_hour7 
 + 1 x_user1_load1_hour6 + 3 x_user1_load2_hour8 + 1 x_user1_load2_hour7 
 + 4 x_user1_load2_hour6 + 2 x_user1_load3_hour8 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour6 + 1 x_user1_load4_hour8 + 2 x_user1_load4_hour7 
 + 3 x_user1_load5_hour8 + 1 x_user1_load5_hour7  <= 4.0
 MaxLoadConnection_user1_9: + 2 x_user1_load0_hour8 + 1 x_user1_load0_hour7 
 + 4 x_user1_load1_hour8 + 1 x_user1_load1_hour7 + 1 x_user1_load2_hour8 
 + 4 x_user1_load2_hour7 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour7 
 + 1 x_user1_load4_hour9 + 2 x_user1_load4_hour8 + 3 x_user1_load5_hour9 
 + 1 x_user1_load5_hour8  <= 4.0
 MaxLoadConnection_user1_10: + 1 x_user1_load0_hour8 + 1 x_user1_load1_hour8 
 + 4 x_user1_load2_hour8 + 1 x_user1_load3_hour8 + 2 x_user1_load4_hour9 
 + 1 x_user1_load5_hour9  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load0_hour8
  x_user1_load1_hour0 x_user1_load1_hour1 x_user1_load1_hour2
  x_user1_load1_hour3 x_user1_load1_hour4 x_user1_load1_hour5
  x_user1_load1_hour6 x_user1_load1_hour7 x_user1_load1_hour8
  x_user1_load2_hour0 x_user1_load2_hour1 x_user1_load2_hour2
  x_user1_load2_hour3 x_user1_load2_hour4 x_user1_load2_hour5
  x_user1_load2_hour6 x_user1_load2_hour7 x_user1_load2_hour8
  x_user1_load3_hour0 x_user1_load3_hour1 x_user1_load3_hour2
  x_user1_load3_hour3 x_user1_load3_hour4 x_user1_load3_hour5
  x_user1_load3_hour6 x_user1_load3_hour7 x_user1_load3_hour8
  x_user1_load4_hour0 x_user1_load4_hour1 x_user1_load4_hour2
  x_user1_load4_hour3 x_user1_load4_hour4 x_user1_load4_hour5
  x_user1_load4_hour6 x_user1_load4_hour7 x_user1_load4_hour8
  x_user1_load4_hour9 x_user1_load5_hour0 x_user1_load5_hour1
  x_user1_load5_hour2 x_user1_load5_hour3 x_user1_load5_hour4
  x_user1_load5_hour5 x_user1_load5_hour6 x_user1_load5_hour7
  x_user1_load5_hour8 x_user1_load5_hour9
General

End