Minimize
 obj: + 149.8 x_user1_load0_hour0 + 189 x_user1_load0_hour1 
 + 207.6 x_user1_load0_hour2 + 231.6 x_user1_load0_hour3 
 + 233 x_user1_load0_hour4 + 229.8 x_user1_load0_hour5 
 + 212 x_user1_load0_hour6 + 189.6 x_user1_load0_hour7 
 + 188.8 x_user1_load1_hour0 + 228.4 x_user1_load1_hour1 
 + 252.8 x_user1_load1_hour2 + 275 x_user1_load1_hour3 
 + 272 x_user1_load1_hour4 + 260.8 x_user1_load1_hour5 
 + 242.4 x_user1_load1_hour6 + 213.8 x_user1_load1_hour7 
 + 180.6 x_user1_load2_hour0 + 225.8 x_user1_load2_hour1 
 + 246.8 x_user1_load2_hour2 + 272.6 x_user1_load2_hour3 
 + 270.2 x_user1_load2_hour4 + 264.6 x_user1_load2_hour5 
 + 244.8 x_user1_load2_hour6 + 215.8 x_user1_load2_hour7 
 + 119 x_user1_load3_hour0 + 152.2 x_user1_load3_hour1 
 + 168.4 x_user1_load3_hour2 + 190.6 x_user1_load3_hour3 
 + 195.8 x_user1_load3_hour4 + 195 x_user1_load3_hour5 
 + 179.2 x_user1_load3_hour6 + 163.4 x_user1_load3_hour7 
 + 107.2 x_user1_load4_hour0 + 126.6 x_user1_load4_hour1 
 + 143.6 x_user1_load4_hour2 + 156.2 x_user1_load4_hour3 
 + 158.4 x_user1_load4_hour4 + 150.2 x_user1_load4_hour5 
 + 139.6 x_user1_load4_hour6 + 126.6 x_user1_load4_hour7 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7  = 1.0
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
 MaxLoadConnection_user1_8: + 1 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 2 x_user1_load1_hour7 + 3 x_user1_load1_hour6 + 1 x_user1_load2_hour7 
 + 3 x_user1_load2_hour6 + 1 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 2 x_user1_load4_hour7 + 1 x_user1_load4_hour6  <= 4.0
 MaxLoadConnection_user1_9: + 2 x_user1_load0_hour7 + 3 x_user1_load1_hour7 
 + 3 x_user1_load2_hour7 + 1 x_user1_load3_hour7 + 1 x_user1_load4_hour7 
  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4 x_user1_load1_hour5 x_user1_load1_hour6
  x_user1_load1_hour7 x_user1_load2_hour0 x_user1_load2_hour1
  x_user1_load2_hour2 x_user1_load2_hour3 x_user1_load2_hour4
  x_user1_load2_hour5 x_user1_load2_hour6 x_user1_load2_hour7
  x_user1_load3_hour0 x_user1_load3_hour1 x_user1_load3_hour2
  x_user1_load3_hour3 x_user1_load3_hour4 x_user1_load3_hour5
  x_user1_load3_hour6 x_user1_load3_hour7 x_user1_load4_hour0
  x_user1_load4_hour1 x_user1_load4_hour2 x_user1_load4_hour3
  x_user1_load4_hour4 x_user1_load4_hour5 x_user1_load4_hour6
  x_user1_load4_hour7
General

End