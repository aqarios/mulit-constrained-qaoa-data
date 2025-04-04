Minimize
 obj: + 147.2 x_user1_load0_hour0 + 183 x_user1_load0_hour1 
 + 205.2 x_user1_load0_hour2 + 229.8 x_user1_load0_hour3 
 + 236.8 x_user1_load0_hour4 + 232.2 x_user1_load0_hour5 
 + 214 x_user1_load0_hour6 + 196.2 x_user1_load0_hour7 
 + 152.8 x_user1_load1_hour0 + 179.6 x_user1_load1_hour1 
 + 208.8 x_user1_load1_hour2 + 230.4 x_user1_load1_hour3 
 + 242.4 x_user1_load1_hour4 + 230.8 x_user1_load1_hour5 
 + 213.6 x_user1_load1_hour6 + 200.8 x_user1_load1_hour7 
 + 170.4 x_user1_load1_hour8 + 169.8 x_user1_load2_hour0 
 + 217.2 x_user1_load2_hour1 + 238.4 x_user1_load2_hour2 
 + 268.4 x_user1_load2_hour3 + 272.2 x_user1_load2_hour4 
 + 270.8 x_user1_load2_hour5 + 249.2 x_user1_load2_hour6 
 + 224.4 x_user1_load2_hour7 + 99 x_user1_load3_hour0 
 + 124 x_user1_load3_hour1 + 137.6 x_user1_load3_hour2 
 + 153.8 x_user1_load3_hour3 + 156.6 x_user1_load3_hour4 
 + 154 x_user1_load3_hour5 + 142 x_user1_load3_hour6 
 + 128.6 x_user1_load3_hour7 + 76.4 x_user1_load4_hour0 
 + 89.8 x_user1_load4_hour1 + 104.4 x_user1_load4_hour2 
 + 115.2 x_user1_load4_hour3 + 121.2 x_user1_load4_hour4 
 + 115.4 x_user1_load4_hour5 + 106.8 x_user1_load4_hour6 
 + 100.4 x_user1_load4_hour7 + 85.2 x_user1_load4_hour8 
 + 88.2 x_user1_load5_hour0 + 115.4 x_user1_load5_hour1 
 + 129.2 x_user1_load5_hour2 + 149.6 x_user1_load5_hour3 
 + 158.6 x_user1_load5_hour4 + 160.2 x_user1_load5_hour5 
 + 146.4 x_user1_load5_hour6 + 137.2 x_user1_load5_hour7 
 + 124.6 x_user1_load5_hour8 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8  = 1.0
 load_load5_scheduled: + 1 x_user1_load5_hour0 + 1 x_user1_load5_hour1 
 + 1 x_user1_load5_hour2 + 1 x_user1_load5_hour3 + 1 x_user1_load5_hour4 
 + 1 x_user1_load5_hour5 + 1 x_user1_load5_hour6 + 1 x_user1_load5_hour7 
 + 1 x_user1_load5_hour8  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 4 x_user1_load2_hour0 + 2 x_user1_load3_hour0 + 1 x_user1_load4_hour0 
 + 3 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 4 x_user1_load1_hour0 + 4 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 2 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour1 + 2 x_user1_load4_hour0 + 3 x_user1_load5_hour1 
 + 1 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 4 x_user1_load1_hour1 
 + 4 x_user1_load2_hour2 + 1 x_user1_load2_hour1 + 2 x_user1_load2_hour0 
 + 2 x_user1_load3_hour2 + 1 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour2 + 2 x_user1_load4_hour1 + 3 x_user1_load5_hour2 
 + 1 x_user1_load5_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 4 x_user1_load1_hour2 
 + 4 x_user1_load2_hour3 + 1 x_user1_load2_hour2 + 2 x_user1_load2_hour1 
 + 2 x_user1_load3_hour3 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
 + 1 x_user1_load4_hour3 + 2 x_user1_load4_hour2 + 3 x_user1_load5_hour3 
 + 1 x_user1_load5_hour2  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 4 x_user1_load1_hour3 
 + 4 x_user1_load2_hour4 + 1 x_user1_load2_hour3 + 2 x_user1_load2_hour2 
 + 2 x_user1_load3_hour4 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
 + 1 x_user1_load4_hour4 + 2 x_user1_load4_hour3 + 3 x_user1_load5_hour4 
 + 1 x_user1_load5_hour3  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 4 x_user1_load1_hour4 
 + 4 x_user1_load2_hour5 + 1 x_user1_load2_hour4 + 2 x_user1_load2_hour3 
 + 2 x_user1_load3_hour5 + 1 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
 + 1 x_user1_load4_hour5 + 2 x_user1_load4_hour4 + 3 x_user1_load5_hour5 
 + 1 x_user1_load5_hour4  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 2 x_user1_load0_hour5 
 + 1 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 4 x_user1_load1_hour5 
 + 4 x_user1_load2_hour6 + 1 x_user1_load2_hour5 + 2 x_user1_load2_hour4 
 + 2 x_user1_load3_hour6 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
 + 1 x_user1_load4_hour6 + 2 x_user1_load4_hour5 + 3 x_user1_load5_hour6 
 + 1 x_user1_load5_hour5  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 1 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 4 x_user1_load1_hour6 
 + 4 x_user1_load2_hour7 + 1 x_user1_load2_hour6 + 2 x_user1_load2_hour5 
 + 2 x_user1_load3_hour7 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
 + 1 x_user1_load4_hour7 + 2 x_user1_load4_hour6 + 3 x_user1_load5_hour7 
 + 1 x_user1_load5_hour6  <= 4.0
 MaxLoadConnection_user1_8: + 2 x_user1_load0_hour7 + 1 x_user1_load0_hour6 
 + 2 x_user1_load1_hour8 + 4 x_user1_load1_hour7 + 1 x_user1_load2_hour7 
 + 2 x_user1_load2_hour6 + 1 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 1 x_user1_load4_hour8 + 2 x_user1_load4_hour7 + 3 x_user1_load5_hour8 
 + 1 x_user1_load5_hour7  <= 4.0
 MaxLoadConnection_user1_9: + 1 x_user1_load0_hour7 + 4 x_user1_load1_hour8 
 + 2 x_user1_load2_hour7 + 1 x_user1_load3_hour7 + 2 x_user1_load4_hour8 
 + 1 x_user1_load5_hour8  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4 x_user1_load1_hour5 x_user1_load1_hour6
  x_user1_load1_hour7 x_user1_load1_hour8 x_user1_load2_hour0
  x_user1_load2_hour1 x_user1_load2_hour2 x_user1_load2_hour3
  x_user1_load2_hour4 x_user1_load2_hour5 x_user1_load2_hour6
  x_user1_load2_hour7 x_user1_load3_hour0 x_user1_load3_hour1
  x_user1_load3_hour2 x_user1_load3_hour3 x_user1_load3_hour4
  x_user1_load3_hour5 x_user1_load3_hour6 x_user1_load3_hour7
  x_user1_load4_hour0 x_user1_load4_hour1 x_user1_load4_hour2
  x_user1_load4_hour3 x_user1_load4_hour4 x_user1_load4_hour5
  x_user1_load4_hour6 x_user1_load4_hour7 x_user1_load4_hour8
  x_user1_load5_hour0 x_user1_load5_hour1 x_user1_load5_hour2
  x_user1_load5_hour3 x_user1_load5_hour4 x_user1_load5_hour5
  x_user1_load5_hour6 x_user1_load5_hour7 x_user1_load5_hour8
General

End