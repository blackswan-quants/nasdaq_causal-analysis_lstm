import pandas as pd
import pytz
from helpermodules import memory_handling as mh

class SpeechProcessor:
    def __init__(self, csv_file, timezone='US/Eastern', words_per_minute=130):
        """
        Initialize the SpeechProcessor with a CSV file, timezone, and words per minute.

        Args:
            csv_file (str): Path to the CSV file containing speech data.
            timezone (str): Timezone for the speech timestamps.
            words_per_minute (int): Average speaking rate in words per minute.
        """
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.est = pytz.timezone(timezone)
        self.words_per_minute = words_per_minute
        self.df['timestamp'] = self.df['date'].apply(lambda x: self.est.localize(x.replace(hour=10, minute=0, second=0)))
        self.df['speech_length_minutes'] = self.df['text'].apply(lambda x: max(1, len(x.split()) / self.words_per_minute))
        self.df_expanded = None  # Initialize df_expanded as None

    @staticmethod
    def split_text_by_minute(text, minutes):
        """
        Split the given text into chunks based on the number of minutes.

        Args:
            text (str): The speech text to be split.
            minutes (int): The number of minutes to split the text into.

        Returns:
            list: A list of text chunks, each representing one minute of speech.
        """
        words = text.split()  # Split the text by whitespace
        words_per_minute = max(1, len(words) // minutes)  # Calculate words per minute
        return [' '.join(words[i:i + words_per_minute]) for i in range(0, len(words), words_per_minute)]

    def process_speeches(self):
        """
        Process the speeches by splitting the text into chunks by minute and expanding the DataFrame.
        """
        self.df['text_by_minute'] = self.df.apply(lambda row: self.split_text_by_minute(row['text'], int(row['speech_length_minutes'])), axis=1)
        df_expanded = self.df.explode('text_by_minute').reset_index(drop=True)
        df_expanded['minute'] = df_expanded.groupby('timestamp').cumcount()
        df_expanded['timestamp'] = df_expanded['timestamp'] + pd.to_timedelta(df_expanded['minute'], unit='m')
        df_expanded = df_expanded.drop(columns=['minute', 'speech_length_minutes'])
        self.df_expanded = df_expanded

    def save_preprocessed_data(self, filename):
        """
        Save the preprocessed DataFrame using a custom PickleHelper class.

        Args:
            filename (str): The filename to save the preprocessed data.
        """
        pickle_helper = mh.PickleHelper(self.df_expanded)
        pickle_helper.pickle_dump(filename)

# Example usage:
# processor = SpeechProcessor('fedspeeches.csv', timezone='US/Pacific', words_per_minute=150)
# processor.process_speeches()
# processor.save_preprocessed_data('fedspeeches_preprocessed')