from recbole.data_post_process.BaseStrategy import BaseStrategy, basic_process_users, basic_process_items
import pandas as pd

class Ml_1MStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        
    @basic_process_users
    def special_process_users(self, users, items, interactions):
        if self.config['attr']['user']['need_item_genre'] == True:
            users, items, interactions = self._add_interested_genre(users, interactions, items)
        if self.config['attr']['user']['need_item_year'] == True:
            users, items, interactions = self._add_interested_year(users, interactions, items)
        if self.config['attr']['user']['need_zipcode_to_region'] == True:
            users, items, interactions = self._add_zipcode_to_region(users, interactions, items)
        users = self._del_user_columns(users)
        return users, items, interactions

    @basic_process_items
    def special_process_items(self, users, items, interactions):
        if self.config['attr']['item']['need_age'] == True:
            users, items, interactions = self._add_age_to_items(users, items, interactions)
        if self.config['attr']['item']['need_gender'] == True:
            users, items, interactions = self._add_gender_to_items(users, items, interactions)
        if self.config['attr']['item']['need_occupation'] == True:
            users, items, interactions = self._add_occupation_to_items(users, items, interactions)
        if self.config['attr']['item']['need_region'] == True:
            users, items, interactions = self._add_region_to_items(users, items, interactions)
        
        items = self._del_item_columns(items)
        return users, items, interactions

    def _add_interested_genre(self, users, interactions, items):
        """
        私有方法：计算并添加用户最感兴趣的 genre 字段
        """
        # 1. 合并interactions和items数据，获取每个用户评价的电影类型
        user_item_genre = interactions.merge(items[[self.config['ITEM_ID_FIELD'], 'genre']], on='item_id', how='left')
        
        # 2. 过滤掉没有类型信息的记录
        user_item_genre = user_item_genre.dropna(subset=['genre'])
        
        # 3. 将电影类型字符串拆分成列表，然后使用explode展开
        user_item_genre['genre_list'] = user_item_genre['genre'].str.split(" ")
        user_genre_expanded = user_item_genre.explode('genre_list')
        
        # 4. 重命名列以便后续处理
        user_genre_expanded = user_genre_expanded.rename(columns={'genre_list': 'single_genre'})
        
        # 5. 统计每个用户每种类型的出现次数
        genre_counts = user_genre_expanded.groupby([self.config['USER_ID_FIELD'], 'single_genre']).size().reset_index(name='count')
        
        # 6. 找出每个用户最感兴趣的类型（出现次数最多的）
        # 使用transform和max来找到每个用户的最大计数
        genre_counts['max_count'] = genre_counts.groupby(self.config['USER_ID_FIELD'])['count'].transform('max')
        
        # 筛选出每个用户计数最大的类型（如果有并列，取第一个）
        user_max_genre = genre_counts[genre_counts['count'] == genre_counts['max_count']].groupby(self.config['USER_ID_FIELD']).first().reset_index()
        
        # 7. 创建用户ID到最感兴趣类型的映射
        user_interested_genre_map = dict(zip(user_max_genre[self.config['USER_ID_FIELD']], user_max_genre['single_genre']))
        
        # 8. 将interested_genre字段添加到users数据框中
        users['interested_genre'] = users[self.config['USER_ID_FIELD']].map(user_interested_genre_map).fillna('Unknown')
        

        return users, items, interactions
    
    def _add_interested_year(self, users, interactions, items):
        """
        私有方法：计算并添加用户最感兴趣的 year 字段
        """
        # 1. 合并interactions和items数据，获取每个用户评价的电影年份
        user_item_year = interactions.merge(items[[self.config['ITEM_ID_FIELD'], 'release_year']], on='item_id', how='left')
        
        # 2. 过滤掉没有年份信息的记录
        user_item_year = user_item_year.dropna(subset=['release_year'])
        
        # 3. 将年份转换为整数类型
        user_item_year['release_year'] = user_item_year['release_year'].astype(int)
        
        # 4. 统计每个用户每种年份的出现次数
        year_counts = user_item_year.groupby([self.config['USER_ID_FIELD'], 'release_year']).size().reset_index(name='count')
        
        # 5. 找出每个用户最感兴趣的年份（出现次数最多的）
        # 使用transform和max来找到每个用户的最大计数
        year_counts['max_count'] = year_counts.groupby(self.config['USER_ID_FIELD'])['count'].transform('max')
        
        # 筛选出每个用户计数最大的年份（如果有并列，取第一个）
        user_max_year = year_counts[year_counts['count'] == year_counts['max_count']].groupby(self.config['USER_ID_FIELD']).first().reset_index()
        
        # 6. 创建用户ID到最感兴趣年份的映射
        user_interested_year_map = dict(zip(user_max_year[self.config['USER_ID_FIELD']], user_max_year['release_year']))
        
        # 7. 将interested_year字段添加到users数据框中
        users['interested_year'] = users[self.config['USER_ID_FIELD']].map(user_interested_year_map).fillna('Unknown')
        
        return users, items, interactions
    
    def _add_zipcode_to_region(self, users, interactions, items):
        """
        私有方法：计算并添加用户最感兴趣的 region 字段
        """
        # 1. 确保zipcode字段存在且为字符串
        if 'zip_code' not in users.columns:
            users['region'] = -1
            return users, items, interactions
        
        # 2. 将zip_code转换为字符串并补齐到5位（如果缺失）
        users['zip_code'] = users['zip_code'].astype(str).str.zfill(5)
        
        # 3. 提取第一个字符作为region，非法zip_code（长度不为5或非数字）设为-1
        def extract_region(zip_code):
            if len(zip_code) != 5 or not zip_code.isdigit():
                return -1
            return int(zip_code[0])
        
        users['region'] = users['zip_code'].apply(extract_region)
        
        
        return users, items, interactions

    def _add_age_to_items(self, users, items, interactions):
        """
        私有方法：计算并添加 item 的 interested_age 字段（评价该 item 的用户群体的年龄众数）
        """
        # 1. 合并 interactions 和 users 数据，获取每个评价用户的年龄
        item_user_age = interactions.merge(users[[self.config['USER_ID_FIELD'], 'age']], on=self.config['USER_ID_FIELD'], how='left')
        
        # 2. 过滤掉没有年龄信息的记录
        item_user_age = item_user_age.dropna(subset=['age'])
        
        # 3. 将年龄转换为整数类型
        item_user_age['age'] = item_user_age['age'].astype(int)
        
        # 4. 统计每个 item 每个年龄的出现次数
        age_counts = item_user_age.groupby([self.config['ITEM_ID_FIELD'], 'age']).size().reset_index(name='count')
        
        # 5. 找出每个 item 出现次数最多的年龄（众数）
        age_counts['max_count'] = age_counts.groupby(self.config['ITEM_ID_FIELD'])['count'].transform('max')
        item_mode_age = age_counts[age_counts['count'] == age_counts['max_count']].groupby(self.config['ITEM_ID_FIELD']).first().reset_index()
        
        # 6. 创建 item_id 到年龄众数的映射
        item_age_map = dict(zip(item_mode_age[self.config['ITEM_ID_FIELD']], item_mode_age['age']))
        
        # 7. 将 interested_age 字段添加到 items 数据框中
        items['interested_age'] = items[self.config['ITEM_ID_FIELD']].map(item_age_map).fillna(-1)
        
        return users, items, interactions

    def _add_gender_to_items(self, users, items, interactions):
        """
        私有方法：计算并添加 item 的 interested_gender 字段（评价该 item 的用户群体的性别众数）
        """
        # 1. 合并 interactions 和 users 数据，获取每个评价用户的性别
        item_user_gender = interactions.merge(users[[self.config['USER_ID_FIELD'], 'gender']], on=self.config['USER_ID_FIELD'], how='left')
        
        # 2. 过滤掉没有性别信息的记录
        item_user_gender = item_user_gender.dropna(subset=['gender'])
        
        # 3. 统计每个 item 每种性别的出现次数
        gender_counts = item_user_gender.groupby([self.config['ITEM_ID_FIELD'], 'gender']).size().reset_index(name='count')
        
        # 4. 找出每个 item 出现次数最多的性别（众数）
        gender_counts['max_count'] = gender_counts.groupby(self.config['ITEM_ID_FIELD'])['count'].transform('max')
        item_mode_gender = gender_counts[gender_counts['count'] == gender_counts['max_count']].groupby(self.config['ITEM_ID_FIELD']).first().reset_index()
        
        # 5. 创建 item_id 到性别众数的映射
        item_gender_map = dict(zip(item_mode_gender[self.config['ITEM_ID_FIELD']], item_mode_gender['gender']))
        
        # 6. 将 gender 字段添加到 items 数据框中
        items['gender'] = items[self.config['ITEM_ID_FIELD']].map(item_gender_map).fillna('Unknown')
        
        items['interested_gender'] = items.pop('gender')  # 将gender字段更名为interested_gender
        
        return users, items, interactions
        
    def _add_occupation_to_items(self, users, items, interactions):
        """
        私有方法：计算并添加 item 的 interested_occupation 字段（评价该 item 的用户群体的职业众数）
        """
        # 1. 合并 interactions 和 users 数据，获取每个评价用户的职业
        item_user_occupation = interactions.merge(users[[self.config['USER_ID_FIELD'], 'occupation']], on=self.config['USER_ID_FIELD'], how='left')
        
        # 2. 过滤掉没有职业信息的记录
        item_user_occupation = item_user_occupation.dropna(subset=['occupation'])
        
        # 3. 统计每个 item 每种职业的出现次数
        occupation_counts = item_user_occupation.groupby([self.config['ITEM_ID_FIELD'], 'occupation']).size().reset_index(name='count')
        
        # 4. 找出每个 item 出现次数最多的职业（众数）
        occupation_counts['max_count'] = occupation_counts.groupby(self.config['ITEM_ID_FIELD'])['count'].transform('max')
        item_mode_occupation = occupation_counts[occupation_counts['count'] == occupation_counts['max_count']].groupby(self.config['ITEM_ID_FIELD']).first().reset_index()
        
        # 5. 创建 item_id 到职业众数的映射
        item_occupation_map = dict(zip(item_mode_occupation[self.config['ITEM_ID_FIELD']], item_mode_occupation['occupation']))
        
        # 6. 将 interested_occupation 字段添加到 items 数据框中
        items['interested_occupation'] = items[self.config['ITEM_ID_FIELD']].map(item_occupation_map).fillna('Unknown')
        
        return users, items, interactions

    def _add_region_to_items(self, users, items, interactions):
        """
        私有方法：计算并添加 item 的 interested_region 字段（评价该 item 的用户群体的地区众数）
        """
        # 1. 合并 interactions 和 users 数据，获取每个评价用户的地区
        item_user_region = interactions.merge(users[[self.config['USER_ID_FIELD'], 'region']], on=self.config['USER_ID_FIELD'], how='left')
        
        # 2. 过滤掉没有地区信息的记录
        item_user_region = item_user_region.dropna(subset=['region'])
        
        # 3. 统计每个 item 每种地区的出现次数
        region_counts = item_user_region.groupby([self.config['ITEM_ID_FIELD'], 'region']).size().reset_index(name='count')
        
        # 4. 找出每个 item 出现次数最多的地区（众数）
        region_counts['max_count'] = region_counts.groupby(self.config['ITEM_ID_FIELD'])['count'].transform('max')
        item_mode_region = region_counts[region_counts['count'] == region_counts['max_count']].groupby(self.config['ITEM_ID_FIELD']).first().reset_index()
        
        # 5. 创建 item_id 到地区众数的映射
        item_region_map = dict(zip(item_mode_region[self.config['ITEM_ID_FIELD']], item_mode_region['region']))
        
        # 6. 将 interested_region 字段添加到 items 数据框中
        items['interested_region'] = items[self.config['ITEM_ID_FIELD']].map(item_region_map).fillna(-1)
        
        return users, items, interactions

    def _del_user_columns(self, users):
        del users['zip_code']
        return users

    def _del_item_columns(self, items):
        del items['movie_title']
        del items['genre']
        return items