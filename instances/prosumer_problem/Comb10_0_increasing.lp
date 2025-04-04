Minimize
 obj: + 127.2 x_user1_load0_hour0 + 137.1 x_user1_load0_hour1 
 + 143.1 x_user1_load0_hour2 + 161.2 x_user1_load0_hour3 
 + 171 x_user1_load0_hour4 + 178.8 x_user1_load0_hour5 
 + 182.6 x_user1_load0_hour6 + 196.5 x_user1_load0_hour7 
 + 212.5 x_user1_load0_hour8 + 152.8 x_user1_load1_hour0 
 + 157.5 x_user1_load1_hour1 + 172.5 x_user1_load1_hour2 
 + 190.3 x_user1_load1_hour3 + 203.9 x_user1_load1_hour4 
 + 206.5 x_user1_load1_hour5 + 215.1 x_user1_load1_hour6 
 + 234.8 x_user1_load1_hour7 + 248.8 x_user1_load1_hour8 
 + 169.2 x_user1_load2_hour0 + 192 x_user1_load2_hour1 
 + 199.5 x_user1_load2_hour2 + 224.2 x_user1_load2_hour3 
 + 227.6 x_user1_load2_hour4 + 242 x_user1_load2_hour5 
 + 255.4 x_user1_load2_hour6 + 268.2 x_user1_load2_hour7 
 + 291.7 x_user1_load2_hour8 + 84.4 x_user1_load3_hour0 
 + 92.7 x_user1_load3_hour1 + 96 x_user1_load3_hour2 
 + 108.4 x_user1_load3_hour3 + 113.6 x_user1_load3_hour4 
 + 119.8 x_user1_load3_hour5 + 123 x_user1_load3_hour6 
 + 131.3 x_user1_load3_hour7 + 142.6 x_user1_load3_hour8 
 + 65.6 x_user1_load4_hour0 + 66 x_user1_load4_hour1 
 + 72.6 x_user1_load4_hour2 + 80.1 x_user1_load4_hour3 
 + 87.5 x_user1_load4_hour4 + 87.9 x_user1_load4_hour5 
 + 90.3 x_user1_load4_hour6 + 99.7 x_user1_load4_hour7 
 + 105.3 x_user1_load4_hour8 + 111.8 x_user1_load4_hour9 
 + 82.8 x_user1_load5_hour0 + 90 x_user1_load5_hour1 
 + 90.3 x_user1_load5_hour2 + 103.8 x_user1_load5_hour3 
 + 112 x_user1_load5_hour4 + 119.2 x_user1_load5_hour5 
 + 117.4 x_user1_load5_hour6 + 126.6 x_user1_load5_hour7 
 + 138.9 x_user1_load5_hour8 + 144.4 x_user1_load5_hour9 

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