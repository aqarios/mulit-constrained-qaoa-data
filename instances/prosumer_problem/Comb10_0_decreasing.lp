Minimize
 obj: + 232.8 x_user1_load0_hour0 + 220.9 x_user1_load0_hour1 
 + 204.9 x_user1_load0_hour2 + 200.8 x_user1_load0_hour3 
 + 189 x_user1_load0_hour4 + 175.2 x_user1_load0_hour5 
 + 157.4 x_user1_load0_hour6 + 149.5 x_user1_load0_hour7 
 + 143.5 x_user1_load0_hour8 + 271.2 x_user1_load1_hour0 
 + 250.5 x_user1_load1_hour1 + 239.5 x_user1_load1_hour2 
 + 231.7 x_user1_load1_hour3 + 220.1 x_user1_load1_hour4 
 + 197.5 x_user1_load1_hour5 + 180.9 x_user1_load1_hour6 
 + 175.2 x_user1_load1_hour7 + 163.2 x_user1_load1_hour8 
 + 296.8 x_user1_load2_hour0 + 290 x_user1_load2_hour1 
 + 268.5 x_user1_load2_hour2 + 263.8 x_user1_load2_hour3 
 + 238.4 x_user1_load2_hour4 + 224 x_user1_load2_hour5 
 + 208.6 x_user1_load2_hour6 + 191.8 x_user1_load2_hour7 
 + 186.3 x_user1_load2_hour8 + 153.6 x_user1_load3_hour0 
 + 147.3 x_user1_load3_hour1 + 136 x_user1_load3_hour2 
 + 133.6 x_user1_load3_hour3 + 124.4 x_user1_load3_hour4 
 + 116.2 x_user1_load3_hour5 + 105 x_user1_load3_hour6 
 + 98.7 x_user1_load3_hour7 + 95.4 x_user1_load3_hour8 
 + 118.4 x_user1_load4_hour0 + 108 x_user1_load4_hour1 
 + 103.4 x_user1_load4_hour2 + 99.9 x_user1_load4_hour3 
 + 96.5 x_user1_load4_hour4 + 86.1 x_user1_load4_hour5 
 + 77.7 x_user1_load4_hour6 + 76.3 x_user1_load4_hour7 
 + 70.7 x_user1_load4_hour8 + 66.2 x_user1_load4_hour9 
 + 159.2 x_user1_load5_hour0 + 152 x_user1_load5_hour1 
 + 137.7 x_user1_load5_hour2 + 136.2 x_user1_load5_hour3 
 + 130 x_user1_load5_hour4 + 122.8 x_user1_load5_hour5 
 + 106.6 x_user1_load5_hour6 + 101.4 x_user1_load5_hour7 
 + 99.1 x_user1_load5_hour8 + 89.6 x_user1_load5_hour9 

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