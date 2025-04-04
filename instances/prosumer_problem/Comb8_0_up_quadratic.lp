Minimize
 obj: + 204.2 x_user1_load0_hour0 + 173 x_user1_load0_hour1 
 + 140.4 x_user1_load0_hour2 + 132.4 x_user1_load0_hour3 
 + 121 x_user1_load0_hour4 + 124.2 x_user1_load0_hour5 
 + 132 x_user1_load0_hour6 + 154.4 x_user1_load0_hour7 
 + 223.2 x_user1_load1_hour0 + 187.6 x_user1_load1_hour1 
 + 159.2 x_user1_load1_hour2 + 151 x_user1_load1_hour3 
 + 140 x_user1_load1_hour4 + 143.2 x_user1_load1_hour5 
 + 161.6 x_user1_load1_hour6 + 192.2 x_user1_load1_hour7 
 + 229.4 x_user1_load2_hour0 + 196.2 x_user1_load2_hour1 
 + 161.2 x_user1_load2_hour2 + 153.4 x_user1_load2_hour3 
 + 139.8 x_user1_load2_hour4 + 145.4 x_user1_load2_hour5 
 + 159.2 x_user1_load2_hour6 + 186.2 x_user1_load2_hour7 
 + 179 x_user1_load3_hour0 + 149.8 x_user1_load3_hour1 
 + 119.6 x_user1_load3_hour2 + 111.4 x_user1_load3_hour3 
 + 102.2 x_user1_load3_hour4 + 103 x_user1_load3_hour5 
 + 104.8 x_user1_load3_hour6 + 122.6 x_user1_load3_hour7 
 + 132.8 x_user1_load4_hour0 + 107.4 x_user1_load4_hour1 
 + 92.4 x_user1_load4_hour2 + 85.8 x_user1_load4_hour3 
 + 81.6 x_user1_load4_hour4 + 79.8 x_user1_load4_hour5 
 + 88.4 x_user1_load4_hour6 + 107.4 x_user1_load4_hour7 

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