import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.log4j.Logger;

class UIDWritable implements Writable {
    private Text uid;
    private IntWritable count;

    public UIDWritable() {
        this.uid = new org.apache.hadoop.io.Text();
        this.count = new IntWritable();
    }
    
    public Text getUID() {
        return this.uid;
    }
    
    public IntWritable getCount() {
        return this.count;
    }
    
    public UIDWritable(Text uid, IntWritable count) {
        this.uid = uid;
        this.count = count;
    }
    
    @Override
    public void readFields(DataInput in) throws IOException {
        uid.readFields(in);
        count.readFields(in);
    }
    
    @Override
    public void write(DataOutput out) throws IOException {
        uid.write(out);
        count.write(out);
    }
}

public class AdsTargeting {
    private static final Logger log = Logger.getLogger(AdsTargeting.class.getName());

    public static class AdsTargetingMapper extends Mapper<LongWritable, Text, Text, UIDWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] fields = value.toString().split("\\s+");
            String uid = fields[0];
            String category = fields[1];
            String[] categories = category.split(",");
            List<List<String>> subsets = subSets(categories);
            UIDWritable writable = new UIDWritable(new Text(uid), ONE);
            
            for (List<String> set : subsets) {
                StringBuilder sb = new StringBuilder();
                for (String s : set) {
                    sb.append(s);
                    sb.append(",");
                }
                sb.deleteCharAt(sb.length()-1);
                String ans = sb.toString();
                context.write(new Text(ans), writable);
            }
        }
        
        // create subsets of original categories.
        public List<List<String>> subSets(String[] categories) {
            List<List<String>> result = new ArrayList<>();
            List<String> tmp = new ArrayList<>();
            backTrack(result, tmp, categories, 0);
            return result;
        }
        
        public void backTrack(List<List<String>> result, List<String> tmp, String[] categories, int idx) {
            if (idx > categories.length) {
                return;
            }
            for (int i=idx; i<categories.length; i++) {
                tmp.add(categories[i]);
                result.add(new ArrayList<String>(tmp));
                backTrack(result, tmp, categories, i+1);
                tmp.remove(tmp.size()-1);
            }
        }
    }
    
    public static class AdsTargetingReducer extends Reducer<Text, UIDWritable, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<UIDWritable> values, Context context) throws IOException, InterruptedException{
            SortedMap<String, Integer> map = new TreeMap<>();
            for (UIDWritable value : values) {
                String uid = value.getUID().toString();
                if (map.containsKey(uid)) {
                    map.put(uid, map.get(uid)+value.getCount().get());
                } else {
                    map.put(uid, value.getCount().get());
                }
            }
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Integer> entry : map.entrySet()) {
                sb.append(entry.getKey());
                sb.append("(");
                sb.append(entry.getValue());
                sb.append(")");
                sb.append(",");
            }
            sb.deleteCharAt(sb.length()-1);
            context.write(key, new Text(sb.toString()));
        }
    }
    
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("<USAGE> <config-options> <INPUT_DIR> <OUTPUT_DIR> ");
            System.exit(0);
        }
        Configuration conf = new Configuration();
        String[] commandArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = new Job(conf, "Ads targeting");
        FileInputFormat.addInputPath(job, new Path(commandArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(commandArgs[1]));
        
        job.setJarByClass(AdsTargeting.class);
        job.setMapperClass(AdsTargeting.AdsTargetingMapper.class);
        job.setReducerClass(AdsTargeting.AdsTargetingReducer.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(UIDWritable.class);
        
        job.setNumReduceTasks(1);        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        int code = job.waitForCompletion(true) ? 0 : 1;
        System.exit(code);
    }
}
