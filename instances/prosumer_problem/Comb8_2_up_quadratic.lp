Minimize
 obj: + 208.7 x_user1_load0_hour0 + 181.6 x_user1_load0_hour1 
 + 150.1 x_user1_load0_hour2 + 140.4 x_user1_load0_hour3 
 + 124.2 x_user1_load0_hour4 + 120 x_user1_load0_hour5 
 + 117 x_user1_load0_hour6 + 126 x_user1_load0_hour7 
 + 148.4 x_user1_load0_hour8 + 168.7 x_user1_load0_hour9 
 + 230.5 x_user1_load1_hour0 + 198.6 x_user1_load1_hour1 
 + 170.4 x_user1_load1_hour2 + 159.2 x_user1_load1_hour3 
 + 141.6 x_user1_load1_hour4 + 135 x_user1_load1_hour5 
 + 139.8 x_user1_load1_hour6 + 153.6 x_user1_load1_hour7 
 + 179.1 x_user1_load1_hour8 + 207.5 x_user1_load1_hour9 
 + 235.6 x_user1_load2_hour0 + 206.6 x_user1_load2_hour1 
 + 172.3 x_user1_load2_hour2 + 162 x_user1_load2_hour3 
 + 142.2 x_user1_load2_hour4 + 138.6 x_user1_load2_hour5 
 + 139.2 x_user1_load2_hour6 + 150 x_user1_load2_hour7 
 + 177.3 x_user1_load2_hour8 + 202.6 x_user1_load2_hour9 
 + 181.8 x_user1_load3_hour0 + 156.6 x_user1_load3_hour1 
 + 127.9 x_user1_load3_hour2 + 118.8 x_user1_load3_hour3 
 + 106.2 x_user1_load3_hour4 + 101.4 x_user1_load3_hour5 
 + 94.8 x_user1_load3_hour6 + 102 x_user1_load3_hour7 
 + 119.5 x_user1_load3_hour8 + 134.8 x_user1_load3_hour9 
 + 136.7 x_user1_load4_hour0 + 113.7 x_user1_load4_hour1 
 + 99.1 x_user1_load4_hour2 + 91 x_user1_load4_hour3 
 + 83.4 x_user1_load4_hour4 + 76.2 x_user1_load4_hour5 
 + 77.4 x_user1_load4_hour6 + 87 x_user1_load4_hour7 
 + 99.1 x_user1_load4_hour8 + 115.7 x_user1_load4_hour9 

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