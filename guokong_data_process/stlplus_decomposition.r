library(stlplus)
library(xts)
library(ggplot2)
library(naniar)
library(readxl)

# site_list = c("1292A", "1203A", "2280A", "2290A", "1150A", "1230A", "2997A", "1997A", "1295A", "1809A", "3046A", "3195A", "1274A", "3187A", "2315A", "1279A", "1169A", "1808A", "1226A", "1291A", "2907A", "2275A", "2298A", "1154A", "2358A", "2284A", "2271A", "2296A", "1229A", "2873A", "1170A", "2316A", "2289A", "2007A", "3207A", "1270A", "1262A", "2318A", "1159A", "1204A", "2382A", "3190A", "2285A", "1257A", "1241A", "1797A", "1252A", "1804A", "2342A", "1166A", "1271A", "2360A", "1290A", "1205A", "2312A", "1796A", "1800A", "1210A", "2299A", "2343A", "2288A", "2286A", "2314A", "3173A", "2994A", "1146A", "2281A", "1278A", "1265A", "1242A", "3002A", "1246A", "1167A", "2287A", "2423A", "2282A", "1221A", "2006A", "1171A", "2346A", "2294A", "1799A", "2311A", "3003A", "2001A", "2273A", "2301A", "2383A", "1256A", "2344A", "1145A", "1141A", "1803A", "1266A", "1147A", "1795A", "2308A", "2357A", "1263A", "1144A", "1233A", "2000A", "1186A", "3164A", "2345A", "2872A", "1294A", "1806A", "1234A", "1298A", "1999A", "2309A", "2278A", "1213A", "2283A", "1264A", "1200A", "1153A", "1240A", "2279A", "2274A", "2306A", "2291A", "1223A", "3191A", "1239A", "2317A", "2005A", "1212A", "1798A", "1165A", "2875A", "1215A", "1148A", "1218A", "2376A", "2379A", "1269A", "1142A", "1228A", "1155A", "2361A", "2870A", "1149A", "2303A", "2277A", "2310A", "2297A", "2292A", "3004A", "1276A", "2307A", "1192A", "1272A", "1267A", "1253A", "2270A", "1196A", "2295A", "1160A", "1235A", "1268A", "1245A", "1162A", "1794A", "3237A", "1236A", "1801A", "2996A", "1211A", "2921A", "2004A", "1255A", "1227A", "1296A", "1232A")

# cat_list = c("CO", "NO2", "O3", "PM10", "PM2.5", "SO2")

# source_dir = "/mnt/d/codes/downloads/datasets/国控站点/stlplus/2014_2020/raw/"
# out_dir = "/mnt/d/codes/downloads/datasets/国控站点/stlplus/2014_2020/stlplus/nofill_single/"

# frequencies <- array(c(1095.75, 2191.5, 4383, 8766), dim=4)


site_list = c("GA40", "WI07", "OH02", "MD08", "MD98", "NY20", "VT99",
       "NY06", "FL96", "MS12")

cat_list = c("GEM", "GOM", "PBM")

source_dir = "/mnt/d/codes/downloads/datasets/北美汞/stlplus/2009_2017/raw/"
out_dir = "/mnt/d/codes/downloads/datasets/北美汞/stlplus/2009_2017/stlplus/nofill_single/"

frequencies <- array(c(547.875, 1095.75, 2191.5, 4383), dim=4)
vals = c(1:4) 


fill = FALSE

for (single_cat in cat_list){
    for (single_site in site_list){
        
        file_name = paste(paste(paste(paste(source_dir, single_cat, sep=""), "_", sep=""), single_site, sep=""), "_nofill.csv", sep="")

        trend_file_out_name = paste(paste(paste(paste(paste(out_dir, "stlplus_trend_nofill_", sep=""), single_cat, sep=""), "_", sep=""), single_site, sep=""), ".csv", sep="")
        seasonal_file_out_name = paste(paste(paste(paste(paste(out_dir, "stlplus_seasonal_nofill_", sep=""), single_cat, sep=""), "_", sep=""), single_site, sep=""), ".csv", sep="")
        remainder_file_out_name = paste(paste(paste(paste(paste(out_dir, "stlplus_remainder_nofill_", sep=""), single_cat, sep=""), "_", sep=""), single_site, sep=""), ".csv", sep="")

        if(file.exists(trend_file_out_name)){
            print("***exists***")
            print(file_name)
            next
        }
        print(file_name)
        
        train_data_station <- read.csv(file_name)
        # 一步是1小时，所以一天是24， 一周是168， 一个月是720， 一年是8766(24*365.25)， 两年是17532
        check <- tryCatch({
            for (i in vals){
                time_series = ts(train_data_station[,2], frequency=frequencies[i])
            
                stl2 = stlplus(time_series, s.window="periodic")
                trend = stl2$data[,"trend"]
                seasonal = stl2$data[, "seasonal"]
                remainder = stl2$data[, "remainder"]

                if(i==1){
                    train_trend_dataframe = data.frame(col1=trend)
                    train_seasonal_dataframe = data.frame(col1=seasonal)
                    train_remainder_dataframe = data.frame(col1=remainder)

                    colnames(train_trend_dataframe) <- c(toString(frequencies[i]))
                    colnames(train_seasonal_dataframe) <- c(toString(frequencies[i]))
                    colnames(train_remainder_dataframe) <- c(toString(frequencies[i]))
                }else{
                    train_trend_dataframe[, paste(frequencies[i])] = trend
                    train_seasonal_dataframe[, paste(frequencies[i])] = seasonal
                    train_remainder_dataframe[, paste(frequencies[i])] = remainder
                    
                }
            }
            write.csv(train_trend_dataframe, trend_file_out_name, row.names=FALSE)
            write.csv(train_seasonal_dataframe, seasonal_file_out_name, row.names=FALSE)
            write.csv(train_remainder_dataframe, remainder_file_out_name, row.names=FALSE)
        },
            warning=function(war){
                print(war)
                print("WARNING")
        },
            error=function(err){
                print("NOT VALID SITE.")
        })
        
    }
}