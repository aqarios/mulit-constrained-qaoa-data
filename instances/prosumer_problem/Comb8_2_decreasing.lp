Minimize
 obj: + 228.7 x_user1_load0_hour0 + 222.6 x_user1_load0_hour1 
 + 205.7 x_user1_load0_hour2 + 203.7 x_user1_load0_hour3 
 + 188.6 x_user1_load0_hour4 + 178.7 x_user1_load0_hour5 
 + 163.7 x_user1_load0_hour6 + 153.6 x_user1_load0_hour7 
 + 150.7 x_user1_load0_hour8 + 138.7 x_user1_load0_hour9 
 + 262.7 x_user1_load1_hour0 + 253 x_user1_load1_hour1 
 + 239.3 x_user1_load1_hour2 + 234.7 x_user1_load1_hour3 
 + 216 x_user1_load1_hour4 + 200.3 x_user1_load1_hour5 
 + 188.7 x_user1_load1_hour6 + 178 x_user1_load1_hour7 
 + 171.3 x_user1_load1_hour8 + 159.7 x_user1_load1_hour9 
 + 263.4 x_user1_load2_hour0 + 257.6 x_user1_load2_hour1 
 + 239 x_user1_load2_hour2 + 236.4 x_user1_load2_hour3 
 + 216.6 x_user1_load2_hour4 + 205 x_user1_load2_hour5 
 + 190.4 x_user1_load2_hour6 + 177.6 x_user1_load2_hour7 
 + 174 x_user1_load2_hour8 + 160.4 x_user1_load2_hour9 
 + 194 x_user1_load3_hour0 + 187.6 x_user1_load3_hour1 
 + 172.4 x_user1_load3_hour2 + 171 x_user1_load3_hour3 
 + 160.6 x_user1_load3_hour4 + 152.4 x_user1_load3_hour5 
 + 137 x_user1_load3_hour6 + 129.6 x_user1_load3_hour7 
 + 127.4 x_user1_load3_hour8 + 117 x_user1_load3_hour9 
 + 153.3 x_user1_load4_hour0 + 143.7 x_user1_load4_hour1 
 + 138 x_user1_load4_hour2 + 134.3 x_user1_load4_hour3 
 + 126.7 x_user1_load4_hour4 + 115 x_user1_load4_hour5 
 + 107.3 x_user1_load4_hour6 + 103.7 x_user1_load4_hour7 
 + 98 x_user1_load4_hour8 + 92.3 x_user1_load4_hour9 

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