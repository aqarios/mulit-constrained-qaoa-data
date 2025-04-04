Minimize
 obj: + 125.3 x_user1_load0_hour0 + 139.4 x_user1_load0_hour1 
 + 142.3 x_user1_load0_hour2 + 160.3 x_user1_load0_hour3 
 + 165.4 x_user1_load0_hour4 + 175.3 x_user1_load0_hour5 
 + 180.3 x_user1_load0_hour6 + 190.4 x_user1_load0_hour7 
 + 207.3 x_user1_load0_hour8 + 215.3 x_user1_load0_hour9 
 + 149.3 x_user1_load1_hour0 + 163 x_user1_load1_hour1 
 + 172.7 x_user1_load1_hour2 + 191.3 x_user1_load1_hour3 
 + 196 x_user1_load1_hour4 + 203.7 x_user1_load1_hour5 
 + 215.3 x_user1_load1_hour6 + 228 x_user1_load1_hour7 
 + 244.7 x_user1_load1_hour8 + 256.3 x_user1_load1_hour9 
 + 146.6 x_user1_load2_hour0 + 164.4 x_user1_load2_hour1 
 + 169 x_user1_load2_hour2 + 189.6 x_user1_load2_hour3 
 + 193.4 x_user1_load2_hour4 + 205 x_user1_load2_hour5 
 + 213.6 x_user1_load2_hour6 + 224.4 x_user1_load2_hour7 
 + 244 x_user1_load2_hour8 + 253.6 x_user1_load2_hour9 
 + 104 x_user1_load3_hour0 + 114.4 x_user1_load3_hour1 
 + 115.6 x_user1_load3_hour2 + 131 x_user1_load3_hour3 
 + 137.4 x_user1_load3_hour4 + 145.6 x_user1_load3_hour5 
 + 147 x_user1_load3_hour6 + 156.4 x_user1_load3_hour7 
 + 170.6 x_user1_load3_hour8 + 177 x_user1_load3_hour9 
 + 86.7 x_user1_load4_hour0 + 90.3 x_user1_load4_hour1 
 + 98 x_user1_load4_hour2 + 107.7 x_user1_load4_hour3 
 + 113.3 x_user1_load4_hour4 + 115 x_user1_load4_hour5 
 + 120.7 x_user1_load4_hour6 + 130.3 x_user1_load4_hour7 
 + 138 x_user1_load4_hour8 + 145.7 x_user1_load4_hour9 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7 
 + 1 x_user1_load0_hour8 + 1 x_user1_load0_hour9  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8 + 1 x_user1_load1_hour9  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7 
 + 1 x_user1_load2_hour8 + 1 x_user1_load2_hour9  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour8 + 1 x_user1_load3_hour9  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8 + 1 x_user1_load4_hour9  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour0 + 1 x_user1_load4_hour0 
  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 1 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 2 x_user1_load1_hour0 + 3 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 3 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour1 + 2 x_user1_load4_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 1 x_user1_load0_hour1 
 + 2 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 2 x_user1_load1_hour1 
 + 3 x_user1_load1_hour0 + 3 x_user1_load2_hour2 + 1 x_user1_load2_hour1 
 + 3 x_user1_load2_hour0 + 3 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour0 + 1 x_user1_load4_hour2 + 2 x_user1_load4_hour1 
 + 1 x_user1_load4_hour0  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 1 x_user1_load0_hour2 
 + 2 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 2 x_user1_load1_hour2 
 + 3 x_user1_load1_hour1 + 3 x_user1_load2_hour3 + 1 x_user1_load2_hour2 
 + 3 x_user1_load2_hour1 + 3 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
 + 1 x_user1_load3_hour1 + 1 x_user1_load4_hour3 + 2 x_user1_load4_hour2 
 + 1 x_user1_load4_hour1  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 1 x_user1_load0_hour3 
 + 2 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 2 x_user1_load1_hour3 
 + 3 x_user1_load1_hour2 + 3 x_user1_load2_hour4 + 1 x_user1_load2_hour3 
 + 3 x_user1_load2_hour2 + 3 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
 + 1 x_user1_load3_hour2 + 1 x_user1_load4_hour4 + 2 x_user1_load4_hour3 
 + 1 x_user1_load4_hour2  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 1 x_user1_load0_hour4 
 + 2 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 2 x_user1_load1_hour4 
 + 3 x_user1_load1_hour3 + 3 x_user1_load2_hour5 + 1 x_user1_load2_hour4 
 + 3 x_user1_load2_hour3 + 3 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour3 + 1 x_user1_load4_hour5 + 2 x_user1_load4_hour4 
 + 1 x_user1_load4_hour3  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 1 x_user1_load0_hour5 
 + 2 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 2 x_user1_load1_hour5 
 + 3 x_user1_load1_hour4 + 3 x_user1_load2_hour6 + 1 x_user1_load2_hour5 
 + 3 x_user1_load2_hour4 + 3 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
 + 1 x_user1_load3_hour4 + 1 x_user1_load4_hour6 + 2 x_user1_load4_hour5 
 + 1 x_user1_load4_hour4  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 1 x_user1_load0_hour6 
 + 2 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 2 x_user1_load1_hour6 
 + 3 x_user1_load1_hour5 + 3 x_user1_load2_hour7 + 1 x_user1_load2_hour6 
 + 3 x_user1_load2_hour5 + 3 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 1 x_user1_load3_hour5 + 1 x_user1_load4_hour7 + 2 x_user1_load4_hour6 
 + 1 x_user1_load4_hour5  <= 4.0
 MaxLoadConnection_user1_8: + 3 x_user1_load0_hour8 + 1 x_user1_load0_hour7 
 + 2 x_user1_load0_hour6 + 2 x_user1_load1_hour8 + 2 x_user1_load1_hour7 
 + 3 x_user1_load1_hour6 + 3 x_user1_load2_hour8 + 1 x_user1_load2_hour7 
 + 3 x_user1_load2_hour6 + 3 x_user1_load3_hour8 + 1 x_user1_load3_hour7 
 + 1 x_user1_load3_hour6 + 1 x_user1_load4_hour8 + 2 x_user1_load4_hour7 
 + 1 x_user1_load4_hour6  <= 4.0
 MaxLoadConnection_user1_9: + 3 x_user1_load0_hour9 + 1 x_user1_load0_hour8 
 + 2 x_user1_load0_hour7 + 2 x_user1_load1_hour9 + 2 x_user1_load1_hour8 
 + 3 x_user1_load1_hour7 + 3 x_user1_load2_hour9 + 1 x_user1_load2_hour8 
 + 3 x_user1_load2_hour7 + 3 x_user1_load3_hour9 + 1 x_user1_load3_hour8 
 + 1 x_user1_load3_hour7 + 1 x_user1_load4_hour9 + 2 x_user1_load4_hour8 
 + 1 x_user1_load4_hour7  <= 4.0
 MaxLoadConnection_user1_10: + 1 x_user1_load0_hour9 + 2 x_user1_load0_hour8 
 + 2 x_user1_load1_hour9 + 3 x_user1_load1_hour8 + 1 x_user1_load2_hour9 
 + 3 x_user1_load2_hour8 + 1 x_user1_load3_hour9 + 1 x_user1_load3_hour8 
 + 2 x_user1_load4_hour9 + 1 x_user1_load4_hour8  <= 4.0
 MaxLoadConnection_user1_11: + 2 x_user1_load0_hour9 + 3 x_user1_load1_hour9 
 + 3 x_user1_load2_hour9 + 1 x_user1_load3_hour9 + 1 x_user1_load4_hour9 
  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load0_hour8
  x_user1_load0_hour9 x_user1_load1_hour0 x_user1_load1_hour1
  x_user1_load1_hour2 x_user1_load1_hour3 x_user1_load1_hour4
  x_user1_load1_hour5 x_user1_load1_hour6 x_user1_load1_hour7
  x_user1_load1_hour8 x_user1_load1_hour9 x_user1_load2_hour0
  x_user1_load2_hour1 x_user1_load2_hour2 x_user1_load2_hour3
  x_user1_load2_hour4 x_user1_load2_hour5 x_user1_load2_hour6
  x_user1_load2_hour7 x_user1_load2_hour8 x_user1_load2_hour9
  x_user1_load3_hour0 x_user1_load3_hour1 x_user1_load3_hour2
  x_user1_load3_hour3 x_user1_load3_hour4 x_user1_load3_hour5
  x_user1_load3_hour6 x_user1_load3_hour7 x_user1_load3_hour8
  x_user1_load3_hour9 x_user1_load4_hour0 x_user1_load4_hour1
  x_user1_load4_hour2 x_user1_load4_hour3 x_user1_load4_hour4
  x_user1_load4_hour5 x_user1_load4_hour6 x_user1_load4_hour7
  x_user1_load4_hour8 x_user1_load4_hour9
General

End